

from dotenv import load_dotenv

from moviepy.editor import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

import cv2
import base64
import io
import openai
import os
import requests
import tempfile

import streamlit as st

#加载 key 等环境常量
load_dotenv()


#视频文件拆帧   流文件转换为视频帧
def video_to_frames(video_file):
    # 上传的视频文件转换存为临时的视频mp4文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_file.read())
        video_filename = tmpfile.name

    #获取视频时长
    video_duration = VideoFileClip(video_filename).duration

    #获取视频数据
    video = cv2.VideoCapture(video_filename)
    base64Frames = []  #存储帧的数组

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)  #拆帧
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))  #转码

    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames, video_filename, video_duration


#识别视频每一帧的内容  根据 prompt 生成文案
def frames_to_story(base64Frames, prompt):
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
                *map(lambda x: {"image": x, "resize": 768},base64Frames[0::25]), #帧数间隔 取关键帧
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "api_key": os.environ["OPENAI_API_KEY"],
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 500,
    }

    result = openai.ChatCompletion.create(**params)
    print(result.choices[0].message.content)
    return result.choices[0].message.content


#文案转为语音 TTS
def text_to_audio(text):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
        json={
            "model": "tts-1",
            "input": text,
            "voice": "nova",  #注意大小写  大写竟然会报错
        },
    )

    if response.status_code != 200:
        raise Exception("Request failed with status code")

    audio_bytes_io = io.BytesIO()

    # 将音频流存入缓存
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio_bytes_io.write(chunk)

    # 定位到音频流起点
    audio_bytes_io.seek(0)

    # 将音频文件存为临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            tmpfile.write(chunk)
        audio_filename = tmpfile.name

    return audio_filename, audio_bytes_io


#将视频个音频文件合并
def merge_audio_video(video_filename, audio_filename, output_filename):
    print("合并 音视频文件 ...")
    print("视频文件 :", video_filename)
    print("音频文件 :", audio_filename)

    # 加载视频文件
    video_clip = VideoFileClip(video_filename)

    # 加载音频文件
    audio_clip = AudioFileClip(audio_filename)

    # 合并音视频文件
    final_clip = video_clip.set_audio(audio_clip)

    # 生成最终的结果文件
    final_clip.write_videofile(
        output_filename, codec='libx264', audio_codec='aac')

    # 关闭流
    video_clip.close()
    audio_clip.close()

    return output_filename

def main():
    st.set_page_config(page_title="GPT 4 Vision + TTS ", page_icon=":smile:")

    st.header("GPT4 Vision + TTS 多模态识别 Demo :smile:")
    uploaded_file = st.file_uploader("加载文件")

    if uploaded_file is not None:
        st.video(uploaded_file)
        prompt = st.text_area(
            "Prompt", value="根据视频元素。 生成具有中国古诗词风格的不少于 15 个字的文案解说。")

    if st.button('生成', type="primary") and uploaded_file is not None:
        with st.spinner('Processing...'):
            base64Frames, video_filename, video_duration = video_to_frames(uploaded_file)

            est_word_count = video_duration * 2
            final_prompt = prompt + f"(这个视频只有 {video_duration} 秒长, 所以需要将文字控制在 {est_word_count} 个字以内)"

            text = frames_to_story(base64Frames, final_prompt)
            st.write(text)

            # 生成文案
            audio_filename, audio_bytes_io = text_to_audio(text)

            # 合并音视频
            output_video_filename = os.path.splitext(video_filename)[0] + '_output.mp4'
            final_video_filename = merge_audio_video(video_filename, audio_filename, output_video_filename)

            # 展示
            st.video(final_video_filename)

            # 清除临时文件
            os.unlink(video_filename)
            os.unlink(audio_filename)
            os.unlink(final_video_filename)

def print_key(name):
    print(os.environ["OPENAI_API_KEY"],)


if __name__ == '__main__':
    main()


