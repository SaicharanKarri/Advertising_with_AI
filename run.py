from module import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True)
    parser.add_argument("--json_path",type=str,required=False)

    args = parser.parse_args()
    video_path = args.video_path
    json_path = args.json_path

    pipe = pipeline(video_path=video_path,advertisements_folder_path="advertisements",json_path=json_path)
    
    obj_det,img_cls,inp_index,out_index = pipe.load_models() #obj_model,interpreter,input_index,output_index
    pipe.load_video(obj_det_model=obj_det,img_cls_model=img_cls,input_index=inp_index,output_index=out_index,roi=0)