# OCR_withCam
Reconhecimento de Caracteres em Fotos de um modulo de Camera no Raspberry Pi3b+

Programa Python rodando em um Raspberry 3B+ acoplado em um modulo de camera, no qual faz a captura da imagem, essa que é reconhecida de acordo com uma base East a fim de localizar e delimitar o território da imagem onde se encontram os caracteres, dessa forma fazemos um corte na imagem de acordo com esse perimetro e por meio da biblioteca OpenCV, realiza-se o tratamento da imagem, buscando o limiar, aplicando desfoque e o laplacian,após, a imagem tratada é direcionada ao programa TesseractOCR, o qual faz o reconhecimento dos caracteres na imagem tratada e retorna o texto reconhecido em string.
