# test_obj_detection_with_app_screen_shots
Various Object Detection models for application screen shots
    

    env common
        pip install uiautomator2==v2.16.23 # in astroboi_m2
        pip install opencv-python==4.5.4.60  # cv2는 반드시 pip 설치


    env for yolo_v5
        conda create -n yolo_v5 python=3.8.13  # Python>=3.8.0 
        conda activate yolo_v5
        
        if silicon mac:
            conda install -c apple tensorflow-deps  # v2.10.0 in mac studio, v2.9.0 in astroboi_m2
            python -m pip install tensorflow-macos  # v2.13.0 in astroboi_m2
            python -m pip install tensorflow-metal  # v1.0.1 in astroboi_m2
            python -m pip install scikit-learn  # v1.3.2 in astroboi_m2
        
        elif windows:
            conda install -c conda-forge tensorflow  # v2.10.0 in mac studio
            conda install -c anaconda scikit-learn  
        
        conda install -c pytorch pytorch=2.0.1  # pytorch, pytorchvision 부터 깔기
        conda install -c pytorch torchvision

        conda install -c anaconda pandas  # v1.4.2 in astroboi_m2
        conda install -c conda-forge pyyaml  #  [주의] yaml이 아니라 pyyaml깐다. pyyaml-5.4.1, yaml-0.2.5 in astroboi_m2
        pip install --force-reinstall charset-normalizer==v3.2.0  # in astroboi_m2
        conda install -c conda-forge tqdm  # 4.66.1  v4.65.0 in astroboi_m2

        conda install -c anaconda seaborn  # v0.11.2 in astroboi_m2  ModuleNotFoundError: No module named 'matplotlib'

