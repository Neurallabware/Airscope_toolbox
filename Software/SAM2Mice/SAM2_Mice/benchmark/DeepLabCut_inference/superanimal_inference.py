import deeplabcut

video_path = r"Y:\LAR\pico\Analysis\tracking\segmentation\benchmark\openfield_three_mice\openfield_three_mouse.mp4"

superanimal_name = "superanimal_topviewmouse"

deeplabcut.video_inference_superanimal(
        videos=[video_path],
        superanimal_name=superanimal_name,
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        video_adapt=False,
        max_individuals=5,
        pseudo_threshold=0.1,
        bbox_threshold=0.9,
        detector_epochs=4,
        pose_epochs=4,
    )