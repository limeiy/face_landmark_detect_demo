#配置
在config里头对tensorflow serving的地址和端口进行配置

#文件用途
net3_conv_augment 是训练脚本，采用卷积神经网络，并且将训练集中所有坐标都标注全的的2000多个图片加了镜像对称，扩充数据集，效果见好
face_landmark_detect_demo   用于测试集图像（16）个批量展示测试结果

#启动服务
将训练好的模型放到tensorflow serving服务器上，做好mapping
docker run -p 8600:8500 -p 8601:8501 \
  --mount type=bind,source=/data/dl_models/tensorflow,target=/models \
  -e MODEL_NAME=tf_face_landmark_detect \
  -t tensorflow/serving

##注意 
1. 8501是restful api用的端口，8500是grpc的端口，本程序用的grpc方式
2. 代码里头的request.model_spec.name = 'tf_face_landmark_detect'  
    这里的名字要跟serving的MODEL_NAME一一对应，否则会找不到
3. 实际模型所在的host文件夹为/data/dl_models/tensorflow/tf_face_landmark_detect/{version}

#运行
运行face_landmark_detect_demo.py脚本即可

#code解析
tensorflow serving服务提供两种访问方式：
1. grpc 方式。采用此方式时，用predict.py里头的do_inference_grpc函数（重命名成do_inference）
2. restful api 方式。采用此方式时，现有代码不用动。
    可通过下述命令验证模型服务是否正确部署：
     curl http://localhost:8501/v1/models/mnist_deep_demo          查看模型状态
     curl http://localhost:8501/v1/models/mnist_deep_demo/metadata 查看模型参数