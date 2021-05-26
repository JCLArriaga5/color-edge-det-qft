import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import cv2

def img_qft(img, mu):
    """
    Obtener la transformada de Fourier para Cuaterniones de una imagen

    Parameters
    ----------
    img : Imagen a color [R, G, B]
    mu : list
      Unidad de cuaternión puro.
      e.g., (i + j + k) / sqrt(3) -> [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]

    Return
    ------
    Fuv : QFT en el dominio de la frecuencia de la imagen de espacio 4-D
    """

    fr = img[:, :, 0]
    fg = img[:, :, 1]
    fb = img[:, :, 2]

    DFTfr = np.fft.fft2(fr)
    DFTfg = np.fft.fft2(fg)
    DFTfb = np.fft.fft2(fb)

    alpha = mu[0]
    betha = mu[1]
    gamma = mu[2]

    Auv = - (alpha * DFTfr[:, :].imag) - (betha * DFTfg[:, :].imag) - (gamma * DFTfb[:, :].imag)
    iBuv = DFTfr[:, :].real + (gamma * DFTfg[:, :].imag) - (betha * DFTfb[:, :].imag)
    jCuv = DFTfg[:, :].real + (alpha * DFTfb[:, :].imag) - (gamma * DFTfr[:, :].imag)
    kDuv = DFTfb[:, :].real + (betha * DFTfr[:, :].imag) - (alpha * DFTfg[:, :].imag)

    Fuv = np.zeros((img.shape[0], img.shape[1], 4))
    Fuv[:, :, 0] = Auv
    Fuv[:, :, 1] = iBuv
    Fuv[:, :, 2] = jCuv
    Fuv[:, :, 3] = kDuv

    return Fuv

def img_iqft(img_qft, mu):
    """
    Transformada inversa de Fourier de la QFT de una imagen.

    Parameters
    ----------
    img_qft : QFT imagen 4-D [Auv, iBuv, jCuv, kDuv]
    mu : list
      Unidad de cuaternión puro.
      e.g., (i + j + k) / sqrt(3) -> [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]

    Returns
    -------
    fmn : Imagen a color
    """

    A = img_qft[:, :, 0]
    B = img_qft[:, :, 1]
    C = img_qft[:, :, 2]
    D = img_qft[:, :, 3]

    IDFTA = np.fft.ifft2(A)
    IDFTB = np.fft.ifft2(B)
    IDFTC = np.fft.ifft2(C)
    IDFTD = np.fft.ifft2(D)

    alpha = mu[0]
    betha = mu[1]
    gamma = mu[2]

    fa = (
        IDFTA[:, :].real - (alpha * IDFTB[:, :].imag) -
        (betha * IDFTC[:, :].imag) - (gamma * IDFTD[:, :].imag)
    )
    fr = (
        IDFTB[:, :].real + (alpha * IDFTA[:, :].imag) +
        (gamma * IDFTC[:, :].imag) - (betha * IDFTD[:, :].imag)
    )
    fg = (
        IDFTC[:, :].real + (betha * IDFTA[:, :].imag) +
        (alpha * IDFTD[:, :].imag) - (gamma * IDFTB[:, :].imag)
    )
    fb = (
        IDFTD[:, :].real + (gamma * IDFTA[:, :].imag) +
        (betha * IDFTB[:, :].imag) - (alpha * IDFTC[:, :].imag)
    )

    fmn = np.zeros((img_qft.shape[0], img_qft.shape[1], 4))
    fmn[:, :, 0] = fa
    fmn[:, :, 1] = fr
    fmn[:, :, 2] = fg
    fmn[:, :, 3] = fb

    return fmn

def sobel_filter_qft(f):
    # sobel in x direction
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    # sobel in y direction
    sobel_y = np.flip(sobel_x.T, axis=0)

    sz_x = (f.shape[0] - sobel_x.shape[0], f.shape[1] - sobel_x.shape[1])
    sobel_x = np.pad(sobel_x, (((sz_x[0] + 1) // 2, sz_x[0] // 2),
                               ((sz_x[1] + 1) // 2, sz_x[1] // 2)), 'constant')
    sobel_x = fftpack.ifftshift(sobel_x)

    sz_y = (f.shape[0] - sobel_y.shape[0], f.shape[1] - sobel_y.shape[1])
    sobel_y = np.pad(sobel_y, (((sz_y[0] + 1) // 2, sz_y[0] // 2),
                               ((sz_y[1] + 1) // 2, sz_y[1] // 2)), 'constant')
    sobel_y = fftpack.ifftshift(sobel_y)

    Gx = np.zeros((f.shape))
    Gy = np.zeros((f.shape))

    for i in range(f.shape[2]):
        if i == 0:
            Gx[:, :, i] = f[:, :, i] * fftpack.fft2(sobel_x).real
            Gy[:, :, i] = f[:, :, i] * fftpack.fft2(sobel_y).real
        else:
            Gx[:, :, i] = f[:, :, i] * fftpack.fft2(sobel_x).imag
            Gy[:, :, i] = f[:, :, i] * fftpack.fft2(sobel_y).imag

    return Gx, Gy

def img_out(G_n, mu):
    out = img_iqft(G_n, mu)

    for d in range(out.shape[2]):
        np.putmask(out[:, :, d], out[:, :, d] < 0, 0)

    return cv2.normalize(out, None, 0, 1, cv2.NORM_MINMAX)

def color_xyedge_det(img, mu=[(1 / np.sqrt(3))] * 3):
    f = img_qft(img, mu)
    Gx, Gy = sobel_filter_qft(f)

    return img_out(Gx, mu), img_out(Gy, mu)

def correlate_qft(img, mu=[(1 / np.sqrt(3))] * 3):
    f = img_qft(img, mu)
    Gx, Gy = sobel_filter_qft(f)

    correlate = np.zeros(f.shape)

    for j in range(f.shape[2]):
        correlate[:, :, j] = Gx[:, :, j] - Gy[:, :, j]

    return img_out(correlate, mu)

if __name__ == '__main__':
    # Read image
    img = plt.imread('./images/shapes.png')

    # Normalize image
    vmin = img.min()
    vmax = img.max()
    img = (np.float32(img) - vmin) / (vmax - vmin)

    # Get a vertical and horizontal sobel filter apply in image
    img_sobelx, img_sobely = color_xyedge_det(img)

    # Show
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1)

    ax1.imshow(img, cmap='gray')
    ax1.set_title('Input image'), ax1.set_xticks([]), ax1.set_yticks([])

    ax2.imshow(img_sobelx[:, :, 1:], cmap='gray')
    ax2.set_title('IQFT Sobel X'), ax2.set_xticks([]), ax2.set_yticks([])

    ax3.imshow(img_sobely[:, :, 1:], cmap='gray')
    ax3.set_title('IQFT Sobel Y'), ax3.set_xticks([]), ax3.set_yticks([])

    plt.savefig('sobel-hv.png', dpi=150, transparent=True)
    plt.show()

    # Combine horizontal and vertical sobel filter using correlation
    correlate = correlate_qft(img)

    plt.figure(figsize=(8, 8))
    plt.title('Sobel H-V Filter Correlation')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(correlate[:, :, 1:], cmap='gray')

    plt.savefig('sobel-correlate.png', dpi=150, transparent=True)
    plt.show()
