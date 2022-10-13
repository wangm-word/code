# -*-coding: utf-8 -*-
#
# import os, sys
#
# sys.path.append(os.getcwd())
# import onnxruntime
# import onnx
# import cv2
# import torch
# import numpy as np
# import torchvision.transforms as transforms
#
#
# class ONNXModel():
#     def __init__(self, onnx_path):
#         """
#         :param onnx_path:
#         """
#         self.onnx_session = onnxruntime.InferenceSession(onnx_path)
#         self.input_name = self.get_input_name(self.onnx_session)
#         self.output_name = self.get_output_name(self.onnx_session)
#         print("input_name:{}".format(self.input_name))
#         print("output_name:{}".format(self.output_name))
#
#     def get_output_name(self, onnx_session):
#         """
#         output_name = onnx_session.get_outputs()[0].name
#         :param onnx_session:
#         :return:
#         """
#         output_name = []
#         for node in onnx_session.get_outputs():
#             output_name.append(node.name)
#         return output_name
#
#     def get_input_name(self, onnx_session):
#         """
#         input_name = onnx_session.get_inputs()[0].name
#         :param onnx_session:
#         :return:
#         """
#         input_name = []
#         for node in onnx_session.get_inputs():
#             input_name.append(node.name)
#         return input_name
#
#     def get_input_feed(self, input_name, image_numpy):
#         """
#         input_feed={self.input_name: image_numpy}
#         :param input_name:
#         :param image_numpy:
#         :return:
#         """
#         input_feed = {}
#         for name in input_name:
#             input_feed[name] = image_numpy
#         return input_feed
#
#     def forward(self, image_numpy):
#         '''
#         # image_numpy = image.transpose(2, 0, 1)
#         # image_numpy = image_numpy[np.newaxis, :]
#         # onnx_session.run([output_name], {input_name: x})
#         # :param image_numpy:
#         # :return:
#         '''
#         # 输入数据的类型必须与模型一致,以下三种写法都是可以的
#         # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
#         # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
#         input_feed = self.get_input_feed(self.input_name, image_numpy)
#         scores, boxes = self.onnx_session.run(self.output_name, input_feed=input_feed)
#         return scores, boxes
#
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
#
# r_model_path = r"D:\AA_smartmore\smartmore_data\A-jiuding\model\cls-4000.onnx"
# # o_model_path = "/home/zigangzhao/DMS/mtcnn-pytorch/test0815/onnx_model/onet.onnx"
#
# img = cv2.imread("./imgcls/Pic_2022_08_07_094855_56.bmp")
# print(img)
# img = cv2.resize(img, (960,150))
# to_tensor = transforms.ToTensor()
# img = to_tensor(img)
# img = img.unsqueeze_(0)
# # rnet1 = ONNXModel(r_model_path)
# # out,b = rnet1.forward(to_numpy(img))
# # print(out)
#
# rnet_session = onnxruntime.InferenceSession(r_model_path)
# # onet_session = onnxruntime.InferenceSession(o_model_path)
# # compute ONNX Runtime output prediction
# inputs = {rnet_session.get_inputs()[0].name: to_numpy(img)}
# outs = rnet_session.run(None, inputs)
# img_out_y = outs
#
# # print("onnx prediction", outs.argmax(axis=1)[0])
#
# print(img_out_y)
#
#
#
# '''
# code by zzg 2021/04/19
# '''
# '''
# # ILSVRC2012_val_00002557.JPEG 289  --mongoose
# '''
# import os, sys
# sys.path.append(os.getcwd())
# import onnxruntime
# import onnx
# import cv2
# import torch
# import torchvision.models as models
# import numpy as np
# import torchvision.transforms as transforms
# from PIL import Image
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
# def get_test_transform():
#     return transforms.Compose([
#         transforms.Resize([224, 224]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
# image = Image.open('./images/ILSVRC2012_val_00002557.JPEG') # 289
#
# img = get_test_transform()(image)
# img = img.unsqueeze_(0) # -> NCHW, 1,3,224,224
# print("input img mean {} and std {}".format(img.mean(), img.std()))
#
# onnx_model_path = "resnet18.onnx"
# pth_model_path = "resnet18.pth"
#
# ## Host GPU pth测试
# resnet18 = models.resnet18()
# net = resnet18
# net.load_state_dict(torch.load(pth_model_path))
# net.eval()
# output = net(img)
#
# print("pth weights", output.detach().cpu().numpy())
# print("HOST GPU prediction", output.argmax(dim=1)[0].item())
#
# ##onnx测试
# resnet_session = onnxruntime.InferenceSession(onnx_model_path)
# #compute ONNX Runtime output prediction
# inputs = {resnet_session.get_inputs()[0].name: to_numpy(img)}
# outs = resnet_session.run(None, inputs)[0])
#
# print("onnx weights", outs)
# print("onnx prediction", outs.argmax(axis=1)[0])


import cv2
import numpy as np
import onnxruntime as rt
import torch
import torch.nn.functional as F

def image_process(image_path):
    mean = np.array([[[0.43320087, 0.43320087, 0.43320087]]])  # 训练的时候用来mean和std
    std = np.array([[[0.29722106, 0.29722106, 0.29722106]]])

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (960, 150))  # (96, 96, 3)

    image = img.astype(np.float32) / 255.0
    image = (image - mean) / std

    image = image.transpose((2, 0, 1))  # (3, 96, 96)
    image = image[np.newaxis, :, :, :]  # (1, 3, 96, 96)

    image = np.array(image, dtype=np.float32)

    return image


def onnx_runtime():
    imgdata = image_process("./imgcls/2-s.bmp")
    model_path=r"cls-5200.onnx"

    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    pred_onnx = sess.run([output_name], {input_name: imgdata})

    print("outputs:")
    # print(np.array(pred_onnx).shape)
    v=np.reshape(np.array(pred_onnx),(1,4))
    # print(v)
    print("softmax之前结果",np.array(pred_onnx))

    print("softmax之后结果",F.softmax(torch.Tensor(v),dim=1))


    ''''
    分类：["hw_fanxiang", "hw_yichang", "mushu_mi", "mushu_shu"]
    '''
onnx_runtime()