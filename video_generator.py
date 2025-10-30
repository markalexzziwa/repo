import os
import tempfile
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

try:
    from moviepy.editor import VideoFileClip, AudioFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        import moviepy
        from moviepy import VideoFileClip, AudioFileClip
        MOVIEPY_AVAILABLE = True
    except ImportError:
        MOVIEPY_AVAILABLE = False
        print("Warning: MoviePy not available. Audio will not be added to videos.")

def get_fonts(size_large=36, size_small=24):
    try:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]
        
        for path in font_paths:
            try:
                font_large = ImageFont.truetype(path, size_large)
                font_small = ImageFont.truetype(path, size_small)
                print(f"Using font: {path}")
                return font_large, font_small
            except:
                continue
        
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        print("Using default font")
        return font_large, font_small
        
    except Exception as e:
        print(f"Font loading error: {e}")
        return ImageFont.load_default(), ImageFont.load_default()

def generate_audio(text, species):
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        audio_path = temp_audio.name
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(audio_path)
        
        return audio_path
        
    except Exception as e:
        print(f"Error generating audio: {e}")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        return temp_audio.name

def split_story_into_chunks(story, words_per_chunk=6):
    words = story.split()
    chunks = []
    
    for i in range(0, len(words), words_per_chunk):
        chunk = ' '.join(words[i:i + words_per_chunk])
        chunks.append(chunk)
    
    return chunks

def create_video_with_transition_images(species, story, image_paths, audio_path, output_path, duration):
    try:
        bird_images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((800, 600), Image.Resampling.LANCZOS)
                bird_images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
        
        if not bird_images:
            raise ValueError("No valid images found for video creation")
        
        caption_chunks = split_story_into_chunks(story, words_per_chunk=6)
        print(f"Story split into {len(caption_chunks)} caption chunks")
        
        canvas_width, canvas_height = 1280, 720
        canvas = Image.new('RGB', (canvas_width, canvas_height), color=(240, 240, 240))
        
        font_large, font_small = get_fonts(36, 24)
        
        image_left = (canvas_width - 800) // 2
        image_top = 50
        
        fps = 24
        total_frames = int(duration * fps)
        
        num_images = len(bird_images)
        frames_per_image = total_frames // num_images
        transition_frames = min(30, frames_per_image // 3)
        
        print(f"Creating video with {num_images} images, {frames_per_image} frames per image")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_width, canvas_height))
        
        for img_index in range(num_images):
            current_img = bird_images[img_index]
            next_img = bird_images[(img_index + 1) % num_images] if num_images > 1 else current_img
            
            for frame_index in range(frames_per_image):
                canvas = Image.new('RGB', (canvas_width, canvas_height), color=(240, 240, 240))
                draw = ImageDraw.Draw(canvas)
                
                if num_images > 1 and frame_index >= (frames_per_image - transition_frames):
                    transition_progress = (frame_index - (frames_per_image - transition_frames)) / transition_frames
                    alpha = transition_progress
                    
                    if alpha < 1.0:
                        current_img_array = np.array(current_img).astype(float)
                        next_img_array = np.array(next_img).astype(float)
                        blended_array = (1 - alpha) * current_img_array + alpha * next_img_array
                        display_img = Image.fromarray(blended_array.astype(np.uint8))
                    else:
                        display_img = next_img
                else:
                    display_img = current_img
                
                canvas.paste(display_img, (image_left, image_top))
                
                title = f"The Story of {species}"
                title_bbox = draw.textbbox((0, 0), title, font=font_large)
                title_width = title_bbox[2] - title_bbox[0]
                title_x = (canvas_width - title_width) // 2
                draw.text((title_x, 10), title, fill=(0, 0, 0), font=font_large)
                
                total_frames_so_far = img_index * frames_per_image + frame_index
                progress = total_frames_so_far / total_frames
                chunk_index = min(int(progress * len(caption_chunks)), len(caption_chunks) - 1)
                
                current_caption = caption_chunks[chunk_index]
                
                caption_y = image_top + 600 + 20
                max_width = canvas_width - 100
                
                words = current_caption.split()
                lines = []
                current_line = []
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    bbox = draw.textbbox((0, 0), test_line, font=font_small)
                    test_width = bbox[2] - bbox[0]
                    
                    if test_width <= max_width:
                        current_line.append(word)
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                caption_bg_height = len(lines) * 40 + 20
                caption_bg = Image.new('RGBA', (canvas_width, caption_bg_height), (255, 255, 255, 200))
                canvas.paste(caption_bg, (0, caption_y - 10), caption_bg)
                
                draw = ImageDraw.Draw(canvas)
                for i, line in enumerate(lines):
                    line_bbox = draw.textbbox((0, 0), line, font=font_small)
                    line_width = line_bbox[2] - line_bbox[0]
                    line_x = (canvas_width - line_width) // 2
                    draw.text((line_x, caption_y + i * 35), line, fill=(0, 0, 0), font=font_small)
                
                if len(caption_chunks) > 1:
                    progress_text = f"Caption {chunk_index + 1}/{len(caption_chunks)}"
                    progress_bbox = draw.textbbox((0, 0), progress_text, font=font_small)
                    progress_x = (canvas_width - progress_bbox[2]) // 2
                    progress_y = caption_y + len(lines) * 35 + 15
                    draw.text((progress_x, progress_y), progress_text, fill=(100, 100, 100), font=font_small)
                
                progress_width = 400
                progress_x = (canvas_width - progress_width) // 2
                progress_y = caption_y + len(lines) * 35 + 45
                
                draw.rectangle([progress_x, progress_y, progress_x + progress_width, progress_y + 8], 
                             fill=(200, 200, 200), outline=(100, 100, 100))
                
                fill_width = int(progress_width * progress)
                if fill_width > 0:
                    draw.rectangle([progress_x, progress_y, progress_x + fill_width, progress_y + 8], 
                                 fill=(76, 175, 80))
                
                frame = np.array(canvas)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
        
        out.release()
        
        if MOVIEPY_AVAILABLE and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            try:
                video_clip = VideoFileClip(output_path)
                audio_clip = AudioFileClip(audio_path)
                
                if audio_clip.duration > duration:
                    audio_clip = audio_clip.subclip(0, duration)
                
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(output_path.replace('.mp4', '_with_audio.mp4'), 
                                         codec='libx264', audio_codec='aac')
                
                os.replace(output_path.replace('.mp4', '_with_audio.mp4'), output_path)
                
            except Exception as e:
                print(f"Warning: Could not add audio to video: {e}")
        elif not MOVIEPY_AVAILABLE:
            print("MoviePy not available - video created without audio")
                
        return True
        
    except Exception as e:
        print(f"Error in transition video creation: {e}")
        return False

def create_ken_burns_effect(species, story, image_paths, audio_path, output_path, duration):
    try:
        bird_images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((1200, 900), Image.Resampling.LANCZOS)
                bird_images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
        
        if not bird_images:
            raise ValueError("No valid images found for video creation")
        
        caption_chunks = split_story_into_chunks(story, words_per_chunk=5)
        print(f"Story split into {len(caption_chunks)} caption chunks")
        
        canvas_width, canvas_height = 1280, 720
        font_large, font_small = get_fonts(32, 20)
        
        fps = 24
        total_frames = int(duration * fps)
        num_images = len(bird_images)
        frames_per_image = total_frames // num_images
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_width, canvas_height))
        
        for img_index in range(num_images):
            current_img = bird_images[img_index]
            img_width, img_height = current_img.size
            
            for frame_index in range(frames_per_image):
                canvas = Image.new('RGB', (canvas_width, canvas_height), color=(0, 0, 0))
                draw = ImageDraw.Draw(canvas)
                
                progress = frame_index / frames_per_image
                
                zoom_start = 1.0
                zoom_end = 1.3
                zoom = zoom_start + (zoom_end - zoom_start) * progress
                
                crop_width = int(canvas_width / zoom)
                crop_height = int(canvas_height / zoom)
                
                pan_x = int((img_width - crop_width) * progress * 0.4)
                pan_y = int((img_height - crop_height) * (1 - progress) * 0.4)
                
                pan_x = min(max(pan_x, 0), img_width - crop_width)
                pan_y = min(max(pan_y, 0), img_height - crop_height)
                
                cropped = current_img.crop((pan_x, pan_y, pan_x + crop_width, pan_y + crop_height))
                display_img = cropped.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                
                canvas.paste(display_img, (0, 0))
                
                total_frames_so_far = img_index * frames_per_image + frame_index
                progress_story = total_frames_so_far / total_frames
                chunk_index = min(int(progress_story * len(caption_chunks)), len(caption_chunks) - 1)
                
                current_caption = caption_chunks[chunk_index]
                
                caption_y = canvas_height - 150
                max_width = canvas_width - 100
                
                words = current_caption.split()
                lines = []
                current_line = []
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    bbox = draw.textbbox((0, 0), test_line, font=font_small)
                    test_width = bbox[2] - bbox[0]
                    
                    if test_width <= max_width:
                        current_line.append(word)
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                overlay_height = max(120, len(lines) * 35 + 40)
                overlay = Image.new('RGBA', (canvas_width, overlay_height), (0, 0, 0, 180))
                canvas.paste(overlay, (0, canvas_height - overlay_height), overlay)
                
                draw = ImageDraw.Draw(canvas)
                
                title = f"The Story of {species}"
                title_bbox = draw.textbbox((0, 0), title, font=font_large)
                title_width = title_bbox[2] - title_bbox[0]
                title_x = (canvas_width - title_width) // 2
                draw.text((title_x, canvas_height - overlay_height + 10), title, fill=(255, 255, 255), font=font_large)
                
                for i, line in enumerate(lines[:2]):
                    line_bbox = draw.textbbox((0, 0), line, font=font_small)
                    line_width = line_bbox[2] - line_bbox[0]
                    line_x = (canvas_width - line_width) // 2
                    draw.text((line_x, caption_y + i * 30), line, fill=(255, 255, 255), font=font_small)
                
                if len(caption_chunks) > 1:
                    progress_text = f"{chunk_index + 1}/{len(caption_chunks)}"
                    progress_bbox = draw.textbbox((0, 0), progress_text, font=font_small)
                    progress_x = canvas_width - progress_bbox[2] - 20
                    progress_y = canvas_height - overlay_height + 15
                    draw.text((progress_x, progress_y), progress_text, fill=(255, 255, 255), font=font_small)
                
                counter_text = f"Image {img_index + 1}/{num_images}"
                counter_bbox = draw.textbbox((0, 0), counter_text, font=font_small)
                counter_x = canvas_width - counter_bbox[2] - 20
                counter_y = 20
                draw.text((counter_x, counter_y), counter_text, fill=(255, 255, 255), font=font_small)
                
                frame = np.array(canvas)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
        
        out.release()
        
        if MOVIEPY_AVAILABLE and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            try:
                video_clip = VideoFileClip(output_path)
                audio_clip = AudioFileClip(audio_path)
                
                if audio_clip.duration > duration:
                    audio_clip = audio_clip.subclip(0, duration)
                
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(output_path.replace('.mp4', '_with_audio.mp4'), 
                                         codec='libx264', audio_codec='aac')
                
                os.replace(output_path.replace('.mp4', '_with_audio.mp4'), output_path)
                
            except Exception as e:
                print(f"Warning: Could not add audio to video: {e}")
        elif not MOVIEPY_AVAILABLE:
            print("MoviePy not available - video created without audio")
                
        return True
        
    except Exception as e:
        print(f"Error in Ken Burns video creation: {e}")
        return False

def create_enhanced_story_video(species, story, image_paths, output_path, duration_per_word=0.4, style='transition'):
    try:
        if not image_paths:
            raise ValueError("No images provided for video creation")
        
        word_count = len(story.split())
        duration = max(word_count * duration_per_word, 12)
        
        print(f"Generating audio for {species}...")
        audio_path = generate_audio(story, species)
        
        print(f"Creating {style} video for {species} with {len(image_paths)} images...")
        
        if style == 'ken_burns':
            success = create_ken_burns_effect(species, story, image_paths, audio_path, output_path, duration)
        else:
            success = create_video_with_transition_images(species, story, image_paths, audio_path, output_path, duration)
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        if success:
            print(f"Enhanced video saved to: {output_path}")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"Error creating enhanced video for {species}: {e}")
        return False
