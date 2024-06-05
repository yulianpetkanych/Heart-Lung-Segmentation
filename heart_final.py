import os
import pickle
import pydicom as pdm
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
import torch

#from pydicom import dcmread
from pydicom.dataset import Dataset
#from pydicom.data import get_testdata_file
from pydicom.pixel_data_handlers.util import apply_modality_lut
import skimage.exposure


def heart_segmentation(root_path, 
                       path_to_save, 
                       organ_num, 
                       path_for_npy, 
                       path_for_jpg, 
                       model_path):
    """
    lung = 0
    heart = 1
    """
    root_path = root_path + "/"
    ids_num_shape={}
    ids = os.listdir(root_path)
    #organ_num = 2



    def get_info_files(ids: list, root_path: str) -> dict:

        """
        Takes: list with ids, and path to dirs with id's images.
        Retruns: dict with shapes and num files for each id image.
        """
        ids_num_shape = {}
        for id_ in ids:
            #files_path = ids
            #file_name = files_path[-1]
            #file_path = os.path.join(root_path + id_ + "/" + file_name)\
            file_path = os.path.join(root_path + id_)
            num_files = len(ids)
            shape = pdm.dcmread(file_path).pixel_array.shape
            ids_num_shape[id_] = (num_files, *shape)
            
        return ids_num_shape

    ids_num_shape = get_info_files(ids, root_path)
    print(ids_num_shape.values())


    def window_linear(arr: "np.ndarray", ds: Dataset, width: float, center: float) -> "np.ndarray":
        center -= 0.5
        width -= 1
        below = arr <= (center - width / 2)
        above = arr > (center + width / 2)
        between = np.logical_and(~below, ~above)

        # Signed
        y_min = -2**(ds.BitsStored - 1)
        y_max = 2**(ds.BitsStored - 1) - 1
        if ds.PixelRepresentation == 0:
            # Unsigned
            y_min = 0
            y_max = 2**ds.BitsStored - 1

        y_min = y_min * ds.RescaleSlope + ds.RescaleIntercept
        y_max = y_max * ds.RescaleSlope + ds.RescaleIntercept
        y_range = y_max - y_min

        arr[below] = y_min
        arr[above] = y_max
        if between.any():
            arr[between] = (
                ((arr[between] - center) / width + 0.5) * y_range + y_min
            )
        arr = skimage.exposure.rescale_intensity(arr, out_range=(0., 255.))
        return arr

    def id_dicom_to_numpy(ids: str,
                    root_path: str,
                    shape: tuple,
                    organ_num: int,
                    path_to_save: str,) -> np.ndarray:
        '''
        save all id dicom data files in tensor.
        '''
        ids = sorted(os.listdir(root_path))
        tensor = np.zeros(shape[ids[0]], dtype=np.float32)
        for i in range(len(ids)):
            file_path = os.path.join(root_path, ids[i])
            ds = pdm.dcmread(file_path)
            arr = ds.pixel_array
            if 'RescaleSlope' not in ds:
                print('Dataset has no Rescale Slope/Intercept!')
            else:
                hu = apply_modality_lut(arr, ds)
                if organ_num == 0:
                    arr = window_linear(hu, ds, 1500, -600 )# 3600, 550)
                else:
                    arr = window_linear(hu, ds, 3600, 550)
            tensor[i, :, :] = arr #dm

        with open(os.path.join(path_to_save, "NPDCM" + '.npy'), 'wb') as f:
            pickle.dump(tensor, f)

        return None

    id_dicom_to_numpy(ids, root_path, ids_num_shape, organ_num, path_for_npy)



    def convert_to_rgb(ids: str,
                    root_path: str,
                    save_path: str,
                    resize: bool = True,
                    new_size: int = 512,
                    mask: bool = False,
                    eps: float = 1e-9):

        # name = id_
        # if mask:
        #      name = id_ + "_mask"
            
        file_path = os.path.join(root_path, "NPDCM" + ".npy")
        with open(file_path, 'rb') as f:
            tensor = pickle.load(f)
            
        n_slices = tensor.shape[0]

        for i in tqdm(range(n_slices)):
            im = tensor[i,:,:]
            #im = (im.astype(np.float16) - im.min())*255.0 / (im.max()-im.min()) + eps  # https://www.kaggle.com/rashmibanthia/dicom-jpg
            im = im.astype(np.uint8)
            if resize:
                im = cv2.resize(im, (new_size, new_size))
            if mask:
                im = cv2.rotate(im, cv2.ROTATE_180)
            new_filename = os.path.splitext(ids[i])[0]
            save_path_ = os.path.join(save_path, new_filename + ".jpg")
            _ = cv2.imwrite(save_path_, im)
                
        return None


    convert_to_rgb(ids = ids,
            root_path = path_for_npy,
            save_path = path_for_jpg, 
            resize = False,
            mask = False)


    from segmentation_models_pytorch import Unet


    def create_dataframe(path_to_images):
        # Get list of image files in the directory
        image_files = [f for f in sorted(os.listdir(path_to_images)) if f.endswith('.jpg')]

        # Create empty lists to store the data
        image_ids = []
        mask_ids = []
        ids = []

        # Populate the lists with data from the image files
        for image_file in image_files:
            image_id = os.path.splitext(image_file)[0]
            mask_id = image_id.replace('.jpg', '_mask.jpg')
            id_ = image_id.split('_')[0]

            image_ids.append(image_file)
            mask_ids.append(mask_id)
            ids.append(id_)

        # Create the DataFrame and return it
        df = pd.DataFrame({
            'ImageId': image_ids,
            'MaskId': mask_ids,
            'Id': ids
        })

        return df

    full_scan_example = create_dataframe(path_for_jpg)
    print("\n6\n")


    state_path = model_path
    model = Unet('efficientnet-b2', encoder_weights="imagenet", classes=3, activation=None)
    model.load_state_dict(torch.load(state_path, map_location=torch.device('cpu')))
    model.eval()

    import torch.nn as nn
    from albumentations import Normalize
    def get_id_predictions(net: nn.Module,
                        ct_scan_id_df: pd.DataFrame,
                        root_imgs_dir: str,
                        treshold: float = 0.3) -> list:

        """
        Factory for getting predictions and storing them and images in lists as uint8 images.
        Params:
            net: model for prediction.
            ct_scan_id_df: df with unique patient id.
            root_imgs_dir: root path for images.
            treshold: threshold for probabilities.
        """
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        images = []
        predictions = []
        net.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", device)
        with torch.no_grad():
            for idx in range(len(ct_scan_id_df)):
                img_name = ct_scan_id_df.loc[idx, "ImageId"]
                path = os.path.join(root_imgs_dir, img_name)

                img_ = cv2.imread(path)
                img = Normalize().apply(img_)
                tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
                prediction = net.forward(tensor.to(device))
                prediction = prediction.cpu().detach().numpy()
                prediction = prediction.squeeze(0).transpose(1, 2, 0)
                prediction = sigmoid(prediction)
                prediction = (prediction >= treshold).astype(np.float32)

                predictions.append((prediction * 255).astype("uint8"))
                images.append(img_)

        return images, predictions

    imgs, predictions = get_id_predictions(net=model,
                                        ct_scan_id_df=full_scan_example,
                                        root_imgs_dir=path_for_jpg)#heart_test_jpg

    def delete_needless_organs(predictions, organ_num):
        new_image = np.array(predictions)
        channels_to_zero = [i for i in range(3) if i != organ_num]
        new_image[:, :, :, channels_to_zero] = 0
        return new_image 


    new_prediction = predictions.copy()
    new_prediction = delete_needless_organs(new_prediction, organ_num)

    import matplotlib.pyplot as plt


    def get_overlaid_masks_on_image(
                    one_slice_image: np.ndarray,
                    one_slice_mask: np.ndarray, 
                    w: float = 512,
                    h: float = 512, 
                    dpi: float = 100,
                    write: bool = False,
                    path_to_save: str = '/content/',
                    name_to_save: str = 'img_name'):
        """overlap masks on image and save this as a new image."""

        path_to_save_ = os.path.join(path_to_save, name_to_save)
        lung, heart, trachea = [one_slice_mask[:, :, i] for i in range(3)]
        figsize = (w / dpi), (h / dpi)
        fig = plt.figure(figsize=(figsize))
        fig.add_axes([0, 0, 1, 1])

        # image
        plt.imshow(one_slice_image, cmap="bone")

        # overlaying segmentation masks
        plt.imshow(np.ma.masked_where(lung == False, lung),
                cmap='cool', alpha=0.3)
        plt.imshow(np.ma.masked_where(heart == False, heart),
                cmap='autumn', alpha=0.3)
        plt.imshow(np.ma.masked_where(trachea == False, trachea),
                cmap='autumn_r', alpha=0.3) 

        plt.axis('off')
        fig.savefig(f"{path_to_save_}.png",bbox_inches='tight', 
                    pad_inches=0.0, dpi=dpi,  format="png")
        if write:
            plt.close()
        else:
            plt.show()

    PATH_TO_SAVE = path_to_save

    if not os.path.exists(PATH_TO_SAVE):
        os.mkdir(PATH_TO_SAVE)
        print(f"Folder {PATH_TO_SAVE} created.")


    [
        get_overlaid_masks_on_image(one_slice_image=image,
                                    one_slice_mask=mask, 
                                    write=True,
                                    path_to_save=PATH_TO_SAVE,
                                    name_to_save= str(i_name)
                                    ) 
        for i_name, (image, mask) in enumerate(zip(imgs, new_prediction))
    ]

# heart_segmentation("/home/petkanychyulian/Documents/HLSegmentation/DICOM_files/",
#                    "/home/petkanychyulian/Documents/HLSegmentation/results/",
#                    2,
                   
#                    )
