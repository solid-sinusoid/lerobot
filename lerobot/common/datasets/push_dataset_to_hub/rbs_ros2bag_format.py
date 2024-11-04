#!/usr/bin/env python
"""
Contains utilities to process raw data format of ros2bag files recorded with Robossembler framework

04.11.2024 @shalenikol release 0.1
"""
from pathlib import Path
import shutil

import json
import cv2 # OpenCV library
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
from rosbags.highlevel import AnyReader

import numpy as np
import torch
from datasets import Dataset, Features, Image, Value, Sequence
from PIL import Image as PILImage

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import calculate_episode_data_index, hf_transform_to_torch
from lerobot.common.datasets.video_utils import VideoFrame

FILE_RESULT = "ros2bag_msg.json"
FILE_FRAMES = "ros2bag_frames.json"
MSG_IMAGE = "sensor_msgs/msg/Image"
MSG_JOINTSTATE = "sensor_msgs/msg/JointState"

def check_format(raw_dir: Path) -> bool:
    ros2bag_paths = list(raw_dir.rglob("*.db3"))
    assert len(ros2bag_paths) > 0

    t_d = {}
    for p in ros2bag_paths:
        t_d[p.parent] = None

    assert len(ros2bag_paths) == len(t_d) # each file in its own folder

def get_episode_from_raw(raw_dir: Path) -> tuple:
    t_dir = raw_dir / "rbs_data"
    if t_dir.is_dir():
        shutil.rmtree(t_dir)  # delete the directory and its contents

    p = []
    br = CvBridge()

    # create reader instance and open for reading
    with AnyReader([raw_dir]) as reader:
        # connections = [x for x in reader.connections if x.topic == TOPIC_FOR_EXPORT]
        # list of all messages
        connections = [x for x in reader.connections]
        cameras = {}
        robots = {}
        msg = []
        i = 0
        # print(f"{len(connections)=}")
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            d = {"timestamp": timestamp}
            if connection.msgtype == MSG_IMAGE:
                cameras[connection.topic] = None # non-repeating value
                d["image"] = f"{connection.topic[1:]}/frame_{i:06}.png"
            elif connection.msgtype == MSG_JOINTSTATE:
                robots[connection.topic] = None # non-repeating value
                msg_data = reader.deserialize(rawdata, connection.msgtype)
                msg_d = {}
                msg_d["name"] = msg_data.name
                msg_d["pos"] = msg_data.position.tolist()
                msg_d["vel"] = msg_data.velocity.tolist()
                msg_d["eff"] = msg_data.effort.tolist()
                d["joint_state"] = msg_d
            msg.append(d)
            i += 1

        cameras = list(cameras.keys())
        robots = list(robots.keys())

        # number of messages in topics
        num_msgs = [0] * len(connections)
        for i,x in enumerate(connections):
            num_msgs[i] = sum(1 for _, _, _ in reader.messages(connections=[x]))
            # create folders for images
            if x.msgtype == MSG_IMAGE:
                out_path = t_dir / x.topic[1:]
                Path(out_path).mkdir(parents=True, exist_ok=True)
                p.append(out_path)
        # number of frames
        num_frames = min(num_msgs)

        all_data = {"ROS2_Bag": str(raw_dir), "cameras": cameras , "robots": robots, "num_frames": num_frames, "msg": msg}

        out_file = raw_dir / FILE_RESULT
        with open(out_file, "w") as fh:
            json.dump(all_data, fh, indent=2)

        i = 0
        frames = []
        frame = {}
        images_found = [False] * len(cameras)
        is_js = False
        for t, timestamp, rawdata in reader.messages(connections=connections):
            msg_data = reader.deserialize(rawdata, t.msgtype)
            if t.msgtype == MSG_IMAGE:
                t_idx = cameras.index(t.topic)
                abs_path = str(p[t_idx])
                images_found[t_idx] = True
                # Convert ROS Image message to OpenCV image
                image = br.imgmsg_to_cv2(msg_data)
                im = f"{abs_path}/frame_{i:06}.png"
                cv2.imwrite(im, image)
                frame[t.topic] = im
                # images.append(im)
            elif t.msgtype == MSG_JOINTSTATE:
                msg_d = {}
                msg_d["name"] = msg_data.name
                msg_d["pos"] = msg_data.position.tolist()
                msg_d["vel"] = msg_data.velocity.tolist()
                msg_d["eff"] = msg_data.effort.tolist()
                frame["joint_state"] = msg_d
                is_js = True
            
            if is_js and all(images_found):
                i += 1
                frame["idx"] = i
                frames.append(frame)
                if i == num_frames: # final
                    break
                frame = {}
                is_js = False
                images_found = [False] * len(cameras)

        all_data = {"ROS2_Bag": str(raw_dir), "cameras": cameras , "robots": robots, "num_frames": num_frames, "frames": frames}

        out_file = raw_dir / FILE_FRAMES
        with open(out_file, "w") as fh:
            json.dump(all_data, fh, indent=2)
    return p, all_data

def load_from_raw(raw_dir: Path, fps: int, episodes: list[int] | None = None):
    if episodes is not None:
        # TODO: add support for multi-episodes.
        raise NotImplementedError()

    ros2bag_paths = list(raw_dir.rglob("*.db3"))
    for p in ros2bag_paths:
        episode_path = p.parent

        ep, all_data = get_episode_from_raw(episode_path)
        num_frames = all_data["num_frames"]

        ep_dicts = []
        pos = []
        vel = []
        eff = []
        for v in all_data["frames"]:
            pos.append(v["joint_state"]["pos"])
            vel.append(v["joint_state"]["vel"])
            eff.append(v["joint_state"]["eff"])

        state = torch.from_numpy(np.array(pos, dtype=np.float32))
        velocity = torch.from_numpy(np.array(vel, dtype=np.float32))
        effort = torch.from_numpy(np.array(eff, dtype=np.float32))

        ep_dict = {}
        ep_idx = 0
        for p in ep:
            e = str(p)
            if "depth" in e: # ignore depth channel, not currently handled
                continue

            camera = e.split('/')[-2]
            img_key = f"observation.images.{camera}"

            image_paths = sorted(p.glob("frame_*.png"))

            # ep_dict["observation.image"] = [PILImage.open(x) for x in image_paths]
            ep_dict[img_key] = [PILImage.open(x) for x in image_paths]

        ep_dict["observation.state"] = state
        ep_dict["observation.velocity"] = velocity
        ep_dict["observation.effort"] = effort

        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps

        ep_dicts.append(ep_dict)
    
    data_dict = concatenate_episodes(ep_dicts)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict

def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()
    # if video:
    #     features["observation.image"] = VideoFrame()
    # else:
    #     features["observation.image"] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["observation.velocity"] = Sequence(
        length=data_dict["observation.velocity"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["observation.effort"] = Sequence(
        length=data_dict["observation.effort"].shape[1], feature=Value(dtype="float32", id=None)
    )

    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    # if video or episodes or encoding is not None:
    #     # TODO: support this
    #     raise NotImplementedError

    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 30

    data_dict = load_from_raw(raw_dir, fps, episodes)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
