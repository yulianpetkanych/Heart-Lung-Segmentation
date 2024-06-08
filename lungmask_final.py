import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from lungmask import LMInferer

def lung_segmentation(root_path, path_to_save):
    ids = sorted(os.listdir(root_path))

    def get_id_imgs_predictions(ids, root_path):
        imgs = []
        predictions = []
        for id in ids:
            path = os.path.join(root_path, id)
            
            dicom_image_itk = sitk.ReadImage(path)
        
            inferer = LMInferer(modelname="R231",
                                modelpath=None, 
                                fillmodel=None, 
                                fillmodel_path=None, 
                                force_cpu=True, 
                                batch_size=20, 
                                volume_postprocessing=True, 
                                tqdm_disable=False,
            )
            segmentation = inferer.apply(dicom_image_itk)
            imgs.append(dicom_image_itk)
            predictions.append(segmentation)
        return imgs, predictions

    imgs, segmentation = get_id_imgs_predictions(ids, root_path)

    for id, img, segment in zip(ids, imgs, segmentation):
        mask = segment[0] != 0
        red_image = np.zeros((512, 512, 3), dtype=np.uint8)
        red_image[mask] = [255, 0, 0]
        red_image[~mask] = [0, 0, 0]
        masks = red_image

        image = sitk.GetArrayFromImage(img)[0]  

        alpha = 0.5 

        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.imshow(masks, cmap='hot', alpha=alpha)  
        plt.axis('off')  
        id_jpg = id.replace(".dcm", ".jpg")
        picture_save_path = os.path.join(path_to_save, id_jpg)
        plt.savefig(picture_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    
