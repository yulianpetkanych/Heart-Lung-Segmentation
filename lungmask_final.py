import os
import numpy as np
import pydicom
from lungmask import mask
import SimpleITK as sitk
import matplotlib.pyplot as plt
from lungmask import LMInferer
import cv2

# import os
# import numpy as np
# import pydicom
# #from lungmask import mask
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
# from lungmask import LMInferer
# import cv2
def lung_segmentation(root_path, path_to_save):
    ids = os.listdir(root_path)

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

        # Створимо новий масив з червоними пікселями
        # Важливо, щоб новий масив був трьохканальним (RGB)
        red_image = np.zeros((512, 512, 3), dtype=np.uint8)

        # Встановимо червоний колір для всіх пікселів, що не чорні
        red_image[mask] = [255, 0, 0]

        # Для чорних пікселів залишимо їх чорними
        red_image[~mask] = [0, 0, 0]
        masks = red_image

        image = sitk.GetArrayFromImage(img)[0]  # завантажте ваше зображення (наприклад, з файлу або іншого джерела)

        alpha = 0.5  # прозорість маски від 0 (прозоре) до 1 (непрозоре)

        # Показ зображень за допомогою matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.imshow(masks, cmap='hot', alpha=alpha)  # Використання 'hot' для відображення маски в червоному кольорі
        plt.axis('off')  # Вимкнути осі
        id_jpg = id.replace(".dcm", ".jpg")
        print(id_jpg)
        print(path_to_save)
        print(os.path.join(path_to_save, id_jpg))
        picture_save_path = os.path.join(path_to_save, id_jpg)
        plt.savefig(picture_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    
