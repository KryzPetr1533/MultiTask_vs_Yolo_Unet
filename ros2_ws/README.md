# Краткое руководство по запуску

1. **Сборка и подключение рабочей области**

    Запуск контейнера:
    ```bash
    python3 d.py -bs
    ```
    Внутри контейнера, находясь в /ros2_ws
   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ```

2. **Запуск всех узлов**

   ```bash
   ros2 launch obstacle_detector pipeline_launch.py
   ```

3. **Переопределение параметров (по желанию)**

   ```bash
   ros2 launch obstacle_detector pipeline_launch.py \
     dataroot:=/path/to/nuimages \
     version:=v1.0-train \
     model_path:=/ros2_ws/yolo_best.pt \
     conf_threshold:=0.3 \
     segmentor.model_path:=/ros2_ws/unet_best.pt \
     segmentor.threshold:=0.6
   ```

---

**Узлы, топики и параметры**

* **image\_retriever**

  * Публикует:

    * `/nuimage/image`
    * `/nuimage/mask`
    * `/nuimage/detections`
  * Параметры: `dataroot`, `version`

* **detector**

  * Подписывается: `/nuimage/image`
  * Публикует: `/nuimage/detections_pred`
  * Параметры: `model_path`, `conf_threshold`

* **segmentor**

  * Подписывается: `/nuimage/image`
  * Публикует: `/nuimage/mask_pred`
  * Параметры: `model_path`, `device`, `threshold`

* **metrics**

  * Подписывается на:

    * `/nuimage/detections`
    * `/nuimage/detections_pred`
  * Логирует среднее IoU по таймстемпу

Просто выполните приведённые команды, чтобы запустить весь конвейер.
