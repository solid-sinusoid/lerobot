Добавлена [процедура](lerobot/common/datasets/push_dataset_to_hub/rbs_ros2bag_format.py) конвертации исходного формата "rbs_ros2bag" в lerobot-датасет.

Пример команды:
```bash
python push_dataset_to_hub.py \
    --raw-dir /home/user/rbs_bag_to_lerobot \
    --local-dir data/lerobot/rbs_tst_episode \
    --repo-id lerobot/rbs_tst_episode \
    --raw-format rbs_ros2bag \
    --push-to-hub 0 \
    --video 0 \
    --force-override 1
```