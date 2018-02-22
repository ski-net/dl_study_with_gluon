[BEGAN]

- Generator / Discriminator  구조
   - Generator 구조

     ![gen](images/dec.png)
     - Upsampling 방법을 통해 이미지를 생성함

   - Discriminator 형태
     ![dis](images/dis.png)
     - 기존 GAN의 경우에는 generator를 통해 생성된 이미지와 원 이미지를 비교하는 형태
     - BEGAN의 경우에는 Encoder -> Decoder를 과정을 거쳐 생성된 이미지와 비교를 수행

- loss 정의
   - WGAN의 개념을 일부 사용
   - 일종의 autoencoder를 통해 산출한 결과와 input data 간의 W-dist의 lower bound를 loss로 정의함
   ![loss](images/loss.png)
   - 해당 loss 개념을 기본으로 하여 generator/ Discriminator loss를 정의함
      - 하나 차별 포인트는 Discriminator loss 계산 시 generator 의 loss 를 다 반영하는 것이 이나라 k 라는 parameter와의 곱을 통해 산출하는 것임
      - k 값은 원 이미지를 일종의 autoencoder 형태로 생성한 loss 값과 generator를 통해 생성된 loss 값을 반영하여 산출함
      ![obj](images/obj.png)
