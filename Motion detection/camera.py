import cv2
import datetime
import requests
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_text(img, text, position, font_path, font_size, font_color): # noqa
    """
    画像に日本語のテキストを描画する。
    img: OpenCVの画像
    text: 描画するテキスト（日本語可）
    position: テキストを描画する位置（x, y）
    font_path: 使用するフォントのパス
    font_size: フォントサイズ
    font_color: フォントの色（B, G, R）
    """
    # OpenCVの画像をPillow形式に変換
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 日本語対応フォントを指定
    font = ImageFont.truetype(font_path, font_size)
    # テキストを描画
    draw.text(position, text, fill=font_color, font=font)
    # Pillow形式の画像をOpenCV形式に変換して返す
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# APIキーの設定
OPENWEATHERMAP_API_KEY = '831d18a3e9f150a39d449e66ffef602c' # noqa
NEWSAPI_API_KEY = 'c60ba235c75e41ceb52aa2413e0ac716'

def get_weather_info(): # noqa
    """OpenWeatherMapから神奈川県の天気情報を取得し、表示用の文字列を生成する"""
    city_id = '1848354'  # 神奈川県のCity ID
    url = f"http://api.openweathermap.org/data/2.5/weather?id={city_id}&appid={OPENWEATHERMAP_API_KEY}&units=metric&lang=ja"# noqa
    response = requests.get(url)
    weather_data = response.json()

    if weather_data['cod'] == 200:
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data['wind']['speed']
        temp_max = weather_data['main']['temp_max']
        temp_min = weather_data['main']['temp_min']
        icon_code = weather_data['weather'][0]['icon']
        weather_info = f"湿度: {humidity}%, 風速: {wind_speed}m/s, 最高気温: {temp_max}°C, 最低気温: {temp_min}°C"# noqa
        return weather_info, icon_code
    else:
        return "天気情報が取得できません", ""

def get_news_headline(): # noqa
    """NewsAPIから猫に関するニュースヘッドラインを取得する"""
    url = f"https://newsapi.org/v2/everything?q=cat&language=ja&apiKey={NEWSAPI_API_KEY}"# noqa
    response = requests.get(url)
    news_data = response.json()

    if news_data['status'] == 'ok' and news_data['totalResults'] > 0:
        return news_data['articles'][0]['title']
    else:
        return "ニュース情報が取得できません"

def show_webcam_with_info_and_motion_detection(): # noqa
    # capture番号1で可能
    cap = cv2.VideoCapture(0)
    last_update_time = 0
    weather_info = ""
    news_headline = ""
    # 背景減算器の初期化
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # 動画保存の設定
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 動体検出
        fgmask = fgbg.apply(frame)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # noqa
        for contour in contours:
            # エリア10000が適切
            if cv2.contourArea(contour) < 10000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        current_time = time.time()
        if current_time - last_update_time > 1800:  # 30分ごとに更新(apiの更新)
            weather_info, icon_code = get_weather_info()
            news_headline = get_news_headline()
            last_update_time = current_time

        # 日本語でのテキスト表示（天気情報、ニュースヘッドライン）
        frame = draw_text(frame, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 30), "/Users/airunrun/Downloads/makinas4 2/Makinas-4-Flat.otf", 24, (255, 255, 255))# noqa
        frame = draw_text(frame, weather_info, (10, 60), "/Users/airunrun/Downloads/makinas4 2/Makinas-4-Flat.otf", 24, (255, 255, 255)) # noqa
        frame = draw_text(frame, news_headline, (10, 90), "/Users/airunrun/Downloads/makinas4 2/Makinas-4-Flat.otf", 24, (255, 255, 255))# noqa

        # 動画にフレームを書き込む
        out.write(frame)

        cv2.imshow('Webcam with Info and Motion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 実行
show_webcam_with_info_and_motion_detection() # noqa
