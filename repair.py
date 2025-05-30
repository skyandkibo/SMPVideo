import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
import multiprocessing
import time
from functools import partial
import signal
import sys
import concurrent.futures  # 添加缺失的导入
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def repair_video(input_path, output_path, preset='veryslow', crf=18):
    """
    使用ffmpeg处理视频，忽略错误并使用H.264重新编码，尽量保持质量
    返回处理是否成功
    """
    cmd = [
        'ffmpeg',
        '-v', 'error',         # 只输出错误信息
        '-i', str(input_path), 
        '-c:v', 'libx264',     # 使用H.264编码
        '-preset', preset,     # 编码速度预设
        '-crf', str(crf),      # 质量设置，值越低质量越高
        '-err_detect', 'ignore_err',  # 忽略错误
        '-fflags', '+genpts+igndts',  # 生成PTS，忽略不连续的时间戳
        '-avoid_negative_ts', '1',
        '-y',                  # 覆盖输出文件
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 验证输出视频是否有效
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # 检查视频是否可播放
            check_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0',
                str(output_path)
            ]
            check_result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if check_result.returncode == 0 and check_result.stdout.strip() and int(check_result.stdout.strip()) > 0:
                return True
        
        # 如果处理失败或输出视频无效，删除输出文件
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    
    except Exception as e:
        logger.error(f"处理视频 {input_path} 时出错: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def process_video(video_file, input_dir, output_dir, preset, crf, timeout=3600):
    """处理单个视频文件的工作函数，用于多进程"""
    try:
        # 保持相对路径结构
        rel_path = video_file.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        # 确保输出文件的父目录存在
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # 如果已存在且有效，跳过处理
        if output_path.exists() and output_path.stat().st_size > 0:
            # 检查视频是否可播放
            check_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0',
                str(output_path)
            ]
            check_result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            
            if check_result.returncode == 0 and check_result.stdout.strip() and int(check_result.stdout.strip()) > 0:
                return (str(rel_path), True, "已存在")
        
        # 修复并保存视频
        start_time = time.time()
        # 使用超时参数调用ffmpeg，避免永久卡住
        cmd = [
            'ffmpeg',
            '-v', 'error',
            '-i', str(video_file), 
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-err_detect', 'ignore_err',
            '-fflags', '+genpts+igndts',
            '-avoid_negative_ts', '1',
            '-y',
            str(output_path)
        ]
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 使用超时执行
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return (str(rel_path), False, f"处理超时（超过{timeout}秒）")
        
        success = False
        if returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # 验证视频
            try:
                check_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'v:0',
                    '-count_packets',
                    '-show_entries', 'stream=nb_read_packets',
                    '-of', 'csv=p=0',
                    str(output_path)
                ]
                check_result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                           text=True, timeout=30)
                
                if (check_result.returncode == 0 and 
                    check_result.stdout.strip() and 
                    int(check_result.stdout.strip()) > 0):
                    success = True
            except (subprocess.SubprocessError, ValueError) as e:
                return (str(rel_path), False, f"验证失败: {str(e)}")
        
        elapsed = time.time() - start_time
        
        if success:
            return (str(rel_path), True, f"{elapsed:.1f}秒")
        else:
            # 失败则删除输出文件
            if os.path.exists(output_path):
                os.remove(output_path)
            return (str(rel_path), False, f"处理失败 ({elapsed:.1f}秒)")
    
    except subprocess.TimeoutExpired:
        return (str(video_file), False, f"处理超时（超过{timeout}秒）")
    except Exception as e:
        return (str(video_file), False, f"异常: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='修复训练集视频并使用H.264编码')
    parser.add_argument('--input_dir', type=str, required=True, help='输入视频目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出视频目录')
    parser.add_argument('--ext', type=str, default='.mp4,.avi,.mov', help='视频文件扩展名，用逗号分隔')
    parser.add_argument('--workers', type=int, default=min(2,max(1, multiprocessing.cpu_count() // 2)), 
                        help='并行处理的工作进程数')
    parser.add_argument('--preset', type=str, default='veryslow', 
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 
                                'slow', 'slower', 'veryslow'], 
                        help='FFmpeg编码预设，速度vs质量平衡')
    parser.add_argument('--crf', type=int, default=18, choices=range(0, 52), 
                        help='FFmpeg CRF值，0-51，值越低质量越高，18接近无损')
    parser.add_argument('--log', type=str, default='', help='日志文件路径，留空则只输出到控制台')
    parser.add_argument('--chunksize', type=int, default=5, help='多进程任务批量大小')
    parser.add_argument('--timeout', type=int, default=3600, help='单个视频处理超时时间(秒)')
    parser.add_argument('--error_log', type=str, default='./error.txt', help='错误日志文件路径')
    parser.add_argument('--succeed_log', type=str, default='./succeed.txt', help='成功日志文件路径')    
    parser.add_argument('--limit', type=int, default=0, help='处理文件的最大数量，0表示不限制')
    parser.add_argument('--skip', type=int, default=0, help='跳过前N个文件')
        
    args = parser.parse_args()
    
    os.remove(args.succeed_log) if os.path.exists(args.succeed_log) else None
    os.remove(args.error_log) if os.path.exists(args.error_log) else None

    # 如果指定了日志文件，添加文件处理程序
    if args.log:
        file_handler = logging.FileHandler(args.log)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 处理多个扩展名
    extensions = [ext.strip() if ext.strip().startswith('.') else f'.{ext.strip()}' 
                 for ext in args.ext.split(',')]
    
    # 获取所有视频文件并按路径名排序
    video_files = []
    for ext in extensions:
        video_files.extend(list(input_dir.glob(f'**/*{ext}')))
    
    video_files.sort()  # 按文件路径名排序
    
    # 应用跳过和限制参数
    if args.skip > 0:
        if args.skip >= len(video_files):
            logger.error(f"跳过数量({args.skip})大于或等于文件总数({len(video_files)})，没有可处理的文件")
            return
        logger.info(f"跳过前 {args.skip} 个文件")
        video_files = video_files[args.skip:]
        
    if args.limit > 0 and args.limit < len(video_files):
        logger.info(f"限制处理文件数量为 {args.limit} (总计 {len(video_files)} 个文件)")
        video_files = video_files[:args.limit]
    
    total_files = len(video_files)
    logger.info(f"找到 {total_files} 个视频文件")
    logger.info(f"使用 {args.workers} 个工作进程进行处理")
    logger.info(f"编码参数: preset={args.preset}, crf={args.crf}")
    
    # 准备多进程处理
    process_func = partial(process_video, 
                          input_dir=input_dir, 
                          output_dir=output_dir,
                          preset=args.preset,
                          crf=args.crf,
                          timeout=args.timeout)
    
    # 设置信号处理和中断标志
    interrupted = False
    
    def signal_handler(sig, frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            logger.warning("\n收到中断信号，停止提交新任务...")
            # 如果用户再次按下Ctrl+C，强制退出
            signal.signal(signal.SIGINT, lambda s, f: os._exit(1))
            signal.signal(signal.SIGTERM, lambda s, f: os._exit(1))
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    start_time = time.time()
    successful_count = 0
    failed_files = []
    
    # 使用 ProcessPoolExecutor 替代 multiprocessing.Pool
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            last_update_time = time.time()
            processed_count = 0
            
            # 使用 tqdm 显示进度
            with tqdm(total=total_files+args.skip, initial=args.skip ,desc="处理视频") as pbar:
                # 分批次提交任务并处理完成的任务
                for i in range(0, len(video_files), args.chunksize):
                    # 检查是否被中断
                    if interrupted:
                        logger.info("检测到中断，停止提交新任务")
                        break
                        
                    # 提交批量任务
                    batch = video_files[i:i + args.chunksize]
                    batch_futures = {executor.submit(process_func, video_file): str(video_file) for video_file in batch}
                    futures.update(batch_futures)
                    
                    # 处理已完成的任务
                    done_futures = list(concurrent.futures.as_completed(futures, timeout=args.timeout))
                    for future in done_futures:
                        try:
                            # 获取结果
                            result = future.result(timeout=args.timeout)  
                            filename, success, message = result
                            
                            if success:
                                successful_count += 1
                                with open(args.succeed_log, 'a') as f:
                                    f.write(f"{filename}\n")
                            else:
                                failed_files.append((filename, message))
                                # 记录错误到专门的错误日志
                                if args.error_log:
                                    with open(args.error_log, 'a') as f:
                                        f.write(f"{filename}: {message}\n")
                            
                        except Exception as exc:
                            # 处理异常情况
                            video_path = futures[future]
                            failed_files.append((video_path, f"异常: {str(exc)}"))
                            if args.error_log:
                                with open(args.error_log, 'a') as f:
                                    f.write(f"{video_path}: 异常: {str(exc)}\n")
                        
                        # 每处理完一个任务就更新计数和进度条
                        processed_count += 1
                        pbar.update(1)
                        pbar.set_postfix({"成功": successful_count, "失败": len(failed_files)})
                        
                        # 从futures中删除已处理的任务
                        del futures[future]
                    
                    # 定期输出进度信息
                    current_time = time.time()
                    if current_time - last_update_time > 30:  # 每30秒输出一次状态
                        elapsed = current_time - start_time
                        avg_time = elapsed / max(1, processed_count)
                        est_remaining = avg_time * (total_files - processed_count)
                        
                        logger.info(f"进度：{processed_count}/{total_files} - "
                                    f"已用时间：{elapsed:.1f}秒, 预计剩余时间：{est_remaining:.1f}秒, "
                                    f"成功：{successful_count}, 失败：{len(failed_files)}")
                        last_update_time = current_time
                
                # 处理剩余的任务
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=args.timeout)
                        filename, success, message = result
                        
                        if success:
                            successful_count += 1
                            with open(args.succeed_log, 'a') as f:
                                f.write(f"{filename}\n")
                        else:
                            failed_files.append((filename, message))
                            if args.error_log:
                                with open(args.error_log, 'a') as f:
                                    f.write(f"{filename}: {message}\n")
                    
                    except Exception as exc:
                        video_path = futures[future]
                        failed_files.append((video_path, f"异常: {str(exc)}"))
                        if args.error_log:
                            with open(args.error_log, 'a') as f:
                                f.write(f"{video_path}: 异常: {str(exc)}\n")
                    
                    # 更新进度条
                    processed_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"成功": successful_count, "失败": len(failed_files)})
                    
                    # 定期输出进度
                    current_time = time.time()
                    if current_time - last_update_time > 30:
                        elapsed = current_time - start_time
                        avg_time = elapsed / max(1, processed_count)
                        est_remaining = avg_time * (total_files - processed_count)
                        
                        logger.info(f"进度：{processed_count}/{total_files} - "
                                    f"已用时间：{elapsed:.1f}秒, 预计剩余时间：{est_remaining:.1f}秒, "
                                    f"成功：{successful_count}, 失败：{len(failed_files)}")
                        last_update_time = current_time
    
    except KeyboardInterrupt:
        # 这个异常处理只是为了捕获信号处理器可能没捕获到的中断
        if not interrupted:
            interrupted = True
            logger.warning("\n用户中断处理！等待当前任务完成...")
    
    # 统计和报告结果
    elapsed_time = time.time() - start_time
    logger.info(f"处理完成！用时: {elapsed_time:.1f}秒")
    logger.info(f"成功处理并保存了 {successful_count} 个视频 (共 {total_files} 个)")
    
    if failed_files:
        failed_count = len(failed_files)
        logger.warning(f"有 {failed_count} 个文件处理失败:")
        # 只显示前10个失败
        for filename, message in failed_files[:10]:
            logger.warning(f" - {filename}: {message}")
        
        if failed_count > 10:
            logger.warning(f" - ... 和其他 {failed_count-10} 个文件")
            
        if args.error_log:
            logger.info(f"完整的错误列表已保存到: {args.error_log}")
    
if __name__ == "__main__":
    main()