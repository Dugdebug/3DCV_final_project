# 第五組 final project
## 主題
Stereo Matching

## 目標
1. 在邊緣裝置上進行Stereo Matching，比較不同方法之間的執行速度與效果。
* 方法一：cv2.StereoSGBM
* 方法二：PSMNet
* 方法三：Real Time PSMNet
2. 嘗試應用Stereo Matching的結果做一個簡易的跟車系統。

## 硬體與執行環境
1. Jetson Nano, 4GB
2. 兩個CSICamera
3. Nvidia l4t ml docker container

## 實驗步驟
1. 資料蒐集：使用 jetson nano 於地下室蒐集左右照片，資料集一是測試演算法效果與執行速度使用，左右各50張照片，資料集二是有機車在前方的照片集，用於跟車系統，左右各600張照片，每組照片間個50ms。 
2.  相機校正：因為原本相機魚眼變形嚴重，所以先對左右攝影機做單邊的攝影機校正，得到左右攝影機的內參和變形係數。接著再對左右攝影機同時做立體校正，得到兩攝影機之間的旋轉與平移矩陣與Rectify矩陣。
3. Stereo Matching：比較三種不同方法，cv2.StereoSGBM, PSMNet, Real Time PSMNet。得到各個方法的執行速度與視插圖結果。
4. 物件偵測：使用yolov5做車牌辨識，在搭配視差圖得到深度結果。

## 實驗結果
1. cv2.StereoSGBM : 進行一張Test Data的街景圖需要0.23秒
2. PSMNet：Out of Memory，沒辦法在Nano 4GB上跑。
3. Real time PSMNet : 進行一張Test Data的街景圖需要15秒。
5. 各個方法得到的視插圖結果有在簡報中： https://docs.google.com/presentation/d/1k4fYwIwmIiKkyKmzBem9rGwF_rqLb-VfmrwJW91x7Ek/edit#slide=id.g130aa137bc0_0_412
6. 跟車系統demo：https://drive.google.com/file/d/1url6YtyXJfRKilMaWDTwz2uPs5jtfqTi/view

## 結論
1. 3D convolution在CPU上跑不動，最開始我們用樹莓派嘗試，完全無法進行。
2. 深度網路要應用在自己的場合上，fine-tune是必須的，我們直接用在KITTI上pre-trian的PSMNet來進行，效果不好。
3. PSMNet跟Real time PSMNet在Nano上沒辦法做到Real time，Real time PSMNet使用的設備是NVIDIA Jetson TX2。
4. Stereo SBGM受到環境影響很大，我們在地下室進行，光源條件不佳，深度的計算很不穩定，

## 心得
1. 胡捷翔：這次的期末真的花了很多時間在思考，我們有什麼樣的資源可以用，我們可以做到怎麼樣子的應用。起初因為對在樹莓派上運算深度網路有興趣，想嘗試看看做不做得到，但後來發現3d Conv光在我的m1 pro上跑都會程式直接被kill掉了，更不用說在樹莓派上跑，沒有cuda的優化根本跑不了。後來再借來了jetson nano來用，光要用熟這台邊緣裝置也花了一點時間。原本想在PSMNet上做finetune也遇到我們算力和電腦容量有限的關係，只好放棄，要載下一整個dataset來做pretrain除了算力，要下載的網路流量也是一大問題。但最後有用傳統的方法做到跟車系統的簡單概念還算是有趣。這次的期末真的學到很多經驗，很多事情都細思極恐，但也很有趣，謝謝教授和助教這學期的努力。

2. 林育銓：這次的期末比較有感的就是說在構思題目的時候都覺得可行，實際開始要做才發現好像不是這麼一回事。一開始我們是想說既然樹莓派沒辦法處理那就試著去改深度網路的架構讓他能夠輕量化，看了一些3D convolution的輕量化論文後想把他套到PSMNet上，但套了才發現這樣Pretrain的weight就不能用，我們得重新訓練，但我們沒有那麼多的運算資源跟時間能去train它。後來才改成現在這樣的題目，但我們最後算出來的深度也不穩定，在地下室這種縱深變化很大的地方，Stereo mathching好像不是太適用。電腦視覺的技術要實際應用確實是一個不容易的事情，也難怪說同樣一個題目，會十幾年內都不斷地有新的論文再推出，不過這也讓人覺得是滿值得去研究的。

## 程式分工:
1. 胡捷翔
* dataset_collection/*
* camera_calibration/*
* get_disparity_YOLO.py

2. 林育銓
* object_detection/*

## 實驗分工：
實驗都是兩人一起完成。