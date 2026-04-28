# Image Stitching #

## 특징 ##
- 3장의 이미지를 정합하여, 하나의 큰 이미지를 생성하는 프로그램이다.
- Planar view가 아닌 Spherical view로 구현하였다.

## 핵심 함수 ##
- detect_and_match() : SIFT 특징점 검출, FLANN 매칭, Homography 계산
- spherical_warp() : 이미지를 spherical view로 변환
- warp_two_images() : 두 이미지를 정합하고 하나의 캔버스에 합성
- crop_black_border() : 검은 여백 제거
- stitch_three_images() : stitching 전체 과정 실행

