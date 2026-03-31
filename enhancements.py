# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

color = ['b', 'g', 'r']


def getColorHist(image):
    """
    This function produces channelwise histogram distribution.
    """
    if len(image.shape) == 3:
        hists = []
        for i, col in enumerate(color):
            hists.append(cv2.calcHist([image], [i], None, [256], [0, 256]))
        return hists
    else:
        hists = []
        i = 0
        hists.append(cv2.calcHist([image], [i], None, [256], [0, 256]))
        return hists


def applyCLAHE(image, display: bool = False):
    """
    CLAHE implementation - 完全兼容Keras ImageDataGenerator.
    image: 从Keras ImageDataGenerator传来的图像，通常是float32(0-1范围)
    returns: 与输入相同类型的图像，应用CLAHE增强
    """
    try:
        # 保存原始数据类型和范围信息
        original_dtype = image.dtype
        is_float_input = original_dtype in [np.float32, np.float64]

        # 1. 转换为uint8进行处理（OpenCV CLAHE要求）
        if is_float_input:
            # float (0-1范围) -> uint8 (0-255范围)
            if image.max() <= 1.0 and image.min() >= 0.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                # 其他float范围，归一化到0-1
                image_norm = (image - image.min()) / (image.max() - image.min())
                image_uint8 = (image_norm * 255).astype(np.uint8)
        elif original_dtype == np.uint8:
            # 已经是uint8，直接使用
            image_uint8 = image
        else:
            # 其他类型，强制转换
            image_uint8 = image.astype(np.uint8)

        # 2. 确保图像是3通道RGB格式
        if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
            # Keras通常使用RGB格式
            image_bw = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        elif len(image_uint8.shape) == 2:
            # 已经是灰度图像
            image_bw = image_uint8
        else:
            print(f"警告: 意外的图像形状 {image_uint8.shape}，返回原图")
            return image

        # 3. 应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=5)
        enhanced_img = clahe.apply(image_bw)

        # 4. 转换回3通道
        final_img_uint8 = np.stack((enhanced_img,) * 3, axis=-1)

        # 5. 转换回原始数据类型
        if is_float_input:
            # uint8 -> float (保持0-1范围)
            final_img = final_img_uint8.astype(np.float32) / 255.0
        else:
            # 保持uint8
            final_img = final_img_uint8

        return final_img

    except Exception as e:
        print(f"CLAHE处理失败: {e}，返回原图")
        return image


def applyHistogramEqualization(image, display: bool = False):
    """
    Applies the histogram equalization to a 3 channel grayscale image.
    If display is true, it wills show the comparison.
    """
    # https://123machinelearn.wordpress.com/2017/12/25/image-enhancement-using-high-frequency-emphasis-filtering-and-histogram-equalization/
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)

    hist, bins = np.histogram(image_bw.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_norm = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img_enhanced = cdf[image]

    if display:
        fig = plt.figure(figsize=(10, 5), dpi=300)
        rows, cols = 2, 2

        # Display the original image
        fig.add_subplot(rows, cols, 1)
        plt.imshow(image, cmap=plt.cm.gray);
        plt.axis('off');
        plt.title("Input Image")

        hists = getColorHist(image)
        fig.add_subplot(rows, cols, 3)
        plt.plot(hists[0], color="b")
        plt.plot(hists[1], color="g")
        plt.plot(hists[2], color="r")
        plt.xlim([0, 256])
        plt.title("Original Histogram")

        # Display the thresholded image
        fig.add_subplot(rows, cols, 2)
        plt.imshow(img_enhanced, cmap=plt.cm.gray);plt.axis('off')
        plt.title("Histogram Equalized Image")

        hists = getColorHist(img_enhanced)
        fig.add_subplot(rows, cols, 4)
        plt.plot(hists[0], color="b")
        plt.plot(hists[1], color="g")
        plt.plot(hists[2], color="r")
        plt.xlim([0, 256])
        plt.title("Equalized Histogram")

        plt.show()

        plt.plot

    print("final shape", img_enhanced.shape)
    return img_enhanced


def applyHFEFilter(image, display: bool = False):
    """
    This function applies the High Frequency Emphasis Filter on an Image.
    if display is true, it will show the comparison before and after the filter operation.
    """
    # https://123machinelearn.wordpress.com/2017/12/25/image-enhancement-using-high-frequency-emphasis-filtering-and-histogram-equalization/
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    npFFT = np.fft.fft2(image_bw)
    npFFTS = np.fft.fftshift(npFFT)

    # High-pass Gaussian filter
    (P, Q) = npFFTS.shape
    H = np.zeros((P, Q))
    D0 = 40
    for u in range(P):
        for v in range(Q):
            H[u, v] = 1.0 - np.exp(- ((u - P / 2.0) ** 2 + (v - Q / 2.0) ** 2) / (2 * (D0 ** 2)))
    k1 = 0.5;
    k2 = 0.80
    HFEfilt = k1 + k2 * H  # Apply High-frequency emphasis

    # Apply HFE filter to FFT of original image
    HFE = HFEfilt * npFFTS

    """
    Implement 2D-FFT algorithm

    Input : Input Image
    Output : 2D-FFT of input image
    """

    def fft2d(image):
        # 1) compute 1d-fft on columns
        fftcols = np.array([np.fft.fft(row) for row in image]).transpose()

        # 2) next, compute 1d-fft on in the opposite direction (for each row) on the resulting values
        return np.array([np.fft.fft(row) for row in fftcols]).transpose()

    # Perform IFFT (implemented here using the np.fft function)
    HFEfinal = (np.conjugate(fft2d(np.conjugate(HFE)))) / (P * Q)

    output = np.sqrt((HFEfinal.real) ** 2 + (HFEfinal.imag) ** 2)
    output = np.array(np.stack((output,) * 3, axis=-1), dtype=np.uint8)

    if display:
        fig = plt.figure(figsize=(10, 5), dpi=300)
        rows, cols = 1, 2

        # Display the original image
        fig.add_subplot(rows, cols, 1)
        plt.imshow(image, cmap=plt.cm.gray);
        plt.axis('off');
        plt.title("Input Image")

        # Display the thresholded image
        fig.add_subplot(rows, cols, 2)
        plt.imshow(output, cmap=plt.cm.gray);
        plt.axis('off')
        plt.title("HF Enhanced Image")
        plt.show()

    print(output.shape)
    return output


if __name__ == "__main__":
    img = cv2.imread("/home/sonymd/Downloads/ChestXray14Data/subset/00000003_005.png")
    applyCLAHE(img, display=True)
    #

    applyHistogramEqualization(img, display=True)

    applyHFEFilter(img, display=True)

    # applyHistogramEqualization(output, display=True)
