import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

def img2uint8(img):
    '''
    Convert image to 8-bit format [0, 255].
    '''

    vmin = img.min()
    vmax = img.max()

    if vmax == vmin:
        return np.uint8(img)

    img = ((img - vmin) / (vmax - vmin)) * 255.0

    return np.uint8(img)

def img_qft(img, mu):
    """
    Obtain the Fourier Transform for Quaternions of an Image.

    Parameters
    ----------
    img : Color image [R, G, B].
    mu : list
      Pure quaternion unit.
      e.g., (i + j + k) / sqrt(3) -> [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]

    Return
    ------
    Fuv : QFT in the frequency domain of the 4-D space image.
    """

    if img.dtype != np.uint8:
        img = img2uint8(img)

    fr = img[:, :, 0]
    fg = img[:, :, 1]
    fb = img[:, :, 2]

    DFTfr = np.fft.fft2(fr)
    DFTfg = np.fft.fft2(fg)
    DFTfb = np.fft.fft2(fb)

    alpha = mu[0]
    betha = mu[1]
    gamma = mu[2]

    Auv = - (alpha * DFTfr.imag) - (betha * DFTfg.imag) - (gamma * DFTfb.imag)
    iBuv = DFTfr.real + (gamma * DFTfg.imag) - (betha * DFTfb.imag)
    jCuv = DFTfg.real + (alpha * DFTfb.imag) - (gamma * DFTfr.imag)
    kDuv = DFTfb.real + (betha * DFTfr.imag) - (alpha * DFTfg.imag)

    return np.stack([Auv, iBuv, jCuv, kDuv], axis=-1)

def img_iqft(img_qft, mu):
    """
    Inverse Fourier transform of the QFT of an image.

    Parameters
    ----------
    img_qft : QFT 4-D image [Auv, iBuv, jCuv, kDuv].
    mu : list
      Pure quaternion unit.
      e.g., (i + j + k) / sqrt(3) -> [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]

    Returns
    -------
    fmn : Color image.
    """

    assert img_qft.shape[2] == 4, "Image is not in qft format"

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
        IDFTA.real - (alpha * IDFTB.imag) -
        (betha * IDFTC.imag) - (gamma * IDFTD.imag)
    )
    fr = (
        IDFTB.real + (alpha * IDFTA.imag) +
        (gamma * IDFTC.imag) - (betha * IDFTD.imag)
    )
    fg = (
        IDFTC.real + (betha * IDFTA.imag) +
        (alpha * IDFTD.imag) - (gamma * IDFTB.imag)
    )
    fb = (
        IDFTD.real + (gamma * IDFTA.imag) +
        (betha * IDFTB.imag) - (alpha * IDFTC.imag)
    )

    return np.stack([fa, fr, fg, fb], axis=-1)

def sobel_filter_qft(f, mu=[(1 / np.sqrt(3))] * 3):
    """
    Vertical and horizontal Sobel filter applying the hypercomplex
    convolution theorem (Quaternion cross multiplication).

    Parameters
    ----------
    f : array-like
        QFT image in the frequency domain.
    mu : list, optional
        Pure quaternion unit axis. Default is the gray axis.

    Returns
    -------
    Gx : array-like
        Horizontal Sobel filter response in the QFT frequency domain.
    Gy : array-like
        Vertical Sobel filter response in the QFT frequency domain.
    """

    # sobel in x direction
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    # sobel in y direction
    sobel_y = np.flip(sobel_x.T, axis=0)

    # Padding (Mismo código que tenías)
    sz_x = (f.shape[0] - sobel_x.shape[0], f.shape[1] - sobel_x.shape[1])
    sobel_x = np.pad(sobel_x, (((sz_x[0] + 1) // 2, sz_x[0] // 2),
                               ((sz_x[1] + 1) // 2, sz_x[1] // 2)), 'constant')
    sobel_x = fftpack.ifftshift(sobel_x)

    sz_y = (f.shape[0] - sobel_y.shape[0], f.shape[1] - sobel_y.shape[1])
    sobel_y = np.pad(sobel_y, (((sz_y[0] + 1) // 2, sz_y[0] // 2),
                               ((sz_y[1] + 1) // 2, sz_y[1] // 2)), 'constant')
    sobel_y = fftpack.ifftshift(sobel_y)

    # Transformada 2D estándar de los filtros
    H_x = fftpack.fft2(sobel_x)
    H_y = fftpack.fft2(sobel_y)

    # Extraer partes reales (R) e imaginarias (I)
    R_x, I_x = H_x.real, H_x.imag
    R_y, I_y = H_y.real, H_y.imag
    
    alpha, betha, gamma = mu[0], mu[1], mu[2]
    
    # Componentes del cuaternión de la imagen (A + Bi + Cj + Dk)
    A = f[:, :, 0]
    B = f[:, :, 1]
    C = f[:, :, 2]
    D = f[:, :, 3]

    def quat_mult(A, B, C, D, R, I):
        """Multiplicación q1 * q2 con el filtro mapeado al eje mu"""
        Xi = I * alpha
        Xj = I * betha
        Xk = I * gamma
        
        # Producto cruzado de cuaterniones
        out_A = A * R - B * Xi - C * Xj - D * Xk
        out_B = A * Xi + B * R + C * Xk - D * Xj
        out_C = A * Xj - B * Xk + C * R + D * Xi
        out_D = A * Xk + B * Xj - C * Xi + D * R
        
        return np.stack([out_A, out_B, out_C, out_D], axis=-1)

    # Aplicar el filtro convolucional en dominio hipercomplejo
    Gx = quat_mult(A, B, C, D, R_x, I_x)
    Gy = quat_mult(A, B, C, D, R_y, I_y)

    return Gx, Gy

def img_out(F, mu=[(1 / np.sqrt(3))] * 3):
    """
    Transforms the filtered frequency domain image back to the spatial domain,
    computes the absolute value, and normalizes it to [0, 255] format.

    Parameters
    ----------
    F : array-like
        The filtered QFT image in the frequency domain (4-D).
    mu : list, optional
        Pure quaternion unit axis. Default is the gray axis.

    Returns
    -------
    out : array-like
        The spatial domain image normalized to 8-bit uint8 format.
    """

    assert F.shape[2] == 4, "Image is not in qft format"

    out = img_iqft(F, mu)
    
    # CRÍTICO: Valor absoluto para recuperar bordes con gradiente negativo
    out = np.abs(out) 

    for d in range(out.shape[2]):
        max_val = np.amax(out[:, :, d])
        if max_val > 0: # Prevenir división por cero si el canal es oscuro
            out[:, :, d] *= 255.0 / max_val
        out[:, :, d] = np.clip(out[:, :, d], 0, 255)

    return np.uint8(out)

def color_xyedge_det(img, mu=[(1 / np.sqrt(3))] * 3):
    """
    Color image recovered once the horizontal and vertical sobel filter is
    applied.

    Parameters
    ----------
    img : array-like
        Color image [R, G, B].
    mu : list, optional
        Pure quaternion unit axis. Default is the gray axis.

    Returns
    -------
    out_Gx : array-like
        Image with the horizontal Sobel filter applied.
    out_Gy : array-like
        Image with the vertical Sobel filter applied.
    """

    f = img_qft(img, mu)
    Gx, Gy = sobel_filter_qft(f, mu)

    return img_out(Gx, mu), img_out(Gy, mu)

def gradient_magnitude_qft(img, mu=[(1 / np.sqrt(3))] * 3):
    """
    Calculates the final gradient magnitude of the edges by combining the
    horizontal and vertical Sobel filter responses in the spatial domain.

    Parameters
    ----------
    img : array-like
        Color image [R, G, B].
    mu : list, optional
        Pure quaternion unit axis. Default is the gray axis.

    Returns
    -------
    magnitude : array-like
        The combined edge magnitude image normalized to 8-bit uint8 format.
    """

    f = img_qft(img, mu)
    Gx, Gy = sobel_filter_qft(f, mu)
    
    # 1. Regresar al dominio espacial en crudo (float)
    out_x = img_iqft(Gx, mu)
    out_y = img_iqft(Gy, mu)
    
    # 2. Magnitud del gradiente euclidiano en el espacio
    magnitude = np.sqrt(out_x**2 + out_y**2)
    
    # 3. Normalizar
    for d in range(magnitude.shape[2]):
        max_val = np.amax(magnitude[:, :, d])
        if max_val > 0:
            magnitude[:, :, d] *= 255.0 / max_val
        magnitude[:, :, d] = np.clip(magnitude[:, :, d], 0, 255)
        
    return np.uint8(magnitude)

if __name__ == '__main__':
    # Read image
    img = plt.imread('./images/Lenna.png')

    # Normalize image
    img = img2uint8(img)

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
    fig.savefig('./images/sobel-hv.png', transparent=True)
    plt.show()

    # Combine horizontal and vertical sobel filter to get gradient magnitude
    grad_mag = gradient_magnitude_qft(img)

    plt.figure(figsize=(8, 8))
    plt.title('Sobel H-V Gradient Magnitude')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grad_mag[:, :, 1:], cmap='gray')
    plt.savefig('./images/sobel-grad-mag.png', transparent=True)
    plt.show()
