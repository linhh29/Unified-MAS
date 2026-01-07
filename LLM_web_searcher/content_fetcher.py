"""
内容获取模块
负责从日志文件读取URL，下载PDF/TXT文件，并读取文件内容
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

from LLM_web_searcher.utils import (
    normalize_arxiv_url,
    sanitize_filename,
    find_pdf_links,
    download_pdf,
    is_github_url,
    parse_github_url,
)

# PDF读取库检查
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        print("[Warning] PyPDF2 or pdfplumber not available. PDF reading will be limited.")


def _is_model_workflow_file(file_name: str, file_path: str) -> bool:
    """
    判断文件是否是模型构建和主体workflow相关的文件。
    
    Args:
        file_name: 文件名
        file_path: 文件路径
        
    Returns:
        是否是模型构建相关文件
    """
    file_name_lower = file_name.lower()
    file_path_lower = file_path.lower()
    
    # 排除测试文件和工具文件
    if (file_name_lower.startswith('test_') or 
        file_name_lower.endswith('_test.py') or
        file_name_lower == 'test.py' or
        file_name_lower in ['utils.py', 'helper.py', 'helpers.py', 'config.py', 'settings.py', '__init__.py'] or
        '/test' in file_path_lower or
        '/tests' in file_path_lower or
        '/__pycache__' in file_path_lower or
        '/.git' in file_path_lower):
        return False
    
    # 优先文件：模型构建和workflow相关的文件名
    priority_names = {
        'model.py', 'models.py', 'modeling.py',
        'train.py', 'training.py', 'trainer.py',
        'main.py', 'run.py', 'pipeline.py', 'workflow.py',
        'build.py', 'construct.py', 'architecture.py',
        'net.py', 'network.py', 'nn.py', 'neural_network.py',
        'app.py', 'server.py', 'api.py', 'service.py', 'handler.py'
    }
    
    if file_name_lower in priority_names:
        return True
    
    # 检查路径深度，放宽条件以包含更多文件
    path_parts = file_path.split('/')
    depth = len(path_parts) - 1
    
    # 根目录下的Python文件（深度为0或1）
    if depth <= 1 and file_name_lower.endswith('.py'):
        return True
    
    # 主要目录下的Python文件（放宽到深度3，并扩展主要目录列表）
    main_dirs = {
        'src', 'core', 'model', 'models', 'main', 'train', 'training', 
        'pipeline', 'workflow', 'app', 'application', 'server', 'backend',
        'api', 'service', 'services', 'handler', 'handlers', 'module', 'modules',
        'lib', 'library', 'framework', 'agent', 'agents', 'task', 'tasks'
    }
    
    # 深度放宽到3，并且包含更多主要目录
    if depth <= 3 and any(part.lower() in main_dirs for part in path_parts[:-1]):
        return True
    
    # 如果文件在根目录或一级子目录，且不是测试文件，也接受（放宽条件）
    if depth <= 2 and file_name_lower.endswith('.py'):
        # 排除明显不是主要代码的目录
        excluded_dirs = {'test', 'tests', 'docs', 'doc', 'examples', 'example', 
                        'notebooks', 'notebook', 'scripts', 'tools', 'tool'}
        if not any(part.lower() in excluded_dirs for part in path_parts[:-1]):
            return True
    
    return False


def _get_python_files_recursive(api_url: str, headers: Dict, timeout: int, max_collect: int = 50, strict_filter: bool = True) -> List[Dict]:
    """
    递归获取仓库中的所有Python文件（模型构建相关）。
    
    Args:
        api_url: GitHub API URL
        headers: HTTP请求头
        timeout: 超时时间
        max_collect: 最多收集的文件数量（用于限制递归深度）
        strict_filter: 是否使用严格的过滤条件
        
    Returns:
        Python文件列表
    """
    python_files = []
    
    def traverse_directory(dir_url: str, depth: int = 0, max_depth: int = 4):
        """递归遍历目录"""
        if depth > max_depth or len(python_files) >= max_collect:
            return
        
        try:
            response = requests.get(dir_url, headers=headers, timeout=timeout)
            # print(response)
            if response.status_code != 200:
                if response.status_code == 404:
                    print(f"    [GitHub] Directory not found: {dir_url}")
                return
            
            contents = response.json()

            if not isinstance(contents, list):
                contents = [contents]
            
            for item in contents:
                if len(python_files) >= max_collect:
                    break
                
                item_type = item.get('type', '')
                item_name = item.get('name', '')
                item_path = item.get('path', '')
                
                if item_type == 'file' and item_name.endswith('.py'):
                    # 检查是否是模型构建相关文件
                    if strict_filter:
                        if _is_model_workflow_file(item_name, item_path):
                            python_files.append(item)
                    else:
                        # 宽松模式：只排除明显的测试文件和工具文件
                        if not (item_name.lower().startswith('test_') or 
                                item_name.lower().endswith('_test.py') or
                                item_name.lower() == 'test.py' or
                                '/test' in item_path.lower() or
                                '/tests' in item_path.lower() or
                                '/__pycache__' in item_path.lower()):
                            python_files.append(item)
                elif item_type == 'dir':
                    # 跳过一些不需要的目录
                    if item_name.lower() not in {'test', 'tests', '__pycache__', '.git', 'docs', 'doc', 'examples', 'example', 'notebooks', 'notebook', '.github'}:
                        dir_api_url = item.get('url', '')
                        if dir_api_url:
                            traverse_directory(dir_api_url, depth + 1, max_depth)
        except Exception as e:
            print(f"    [GitHub] Error traversing directory {dir_url}: {e}")
    
    traverse_directory(api_url)
    return python_files


def extract_github_code(
    url: str,
    save_dir: Path,
    title: str,
    headers: Dict,
    timeout: int = 30,
    max_files: int = 10,
    github_token: Optional[str] = None,
) -> Optional[Path]:
    """
    从GitHub URL提取模型构建相关的Python代码文件。
    
    Args:
        url: GitHub URL
        save_dir: 保存目录
        title: 文件标题
        headers: HTTP请求头
        timeout: 超时时间
        max_files: 最多提取的文件数量
        github_token: GitHub Personal Access Token（可选，但强烈建议提供以提高速率限制）
        
    Returns:
        保存的文件路径，如果失败返回None
    """
    try:
        github_info = parse_github_url(url)
        if not github_info:
            print(f"  [GitHub] Failed to parse GitHub URL: {url}")
            return None
        
        owner = github_info['owner']
        repo = github_info['repo']
        branch = github_info['branch']
        path = github_info['path']
        file_type = github_info['file_type']
        
        # 为 GitHub API 请求准备 headers
        github_headers = headers.copy()
        github_headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if github_token:
            github_headers["Authorization"] = f"Bearer {github_token}"
            print(f"  [GitHub] Using GitHub token for authentication")
        else:
            print(f"  [GitHub] Warning: No GitHub token provided. Rate limits may be lower.")
        
        extracted_code = []
        
        if file_type == 'file' and path:
            # 单个文件页面，如果是Python文件则直接提取
            if path.endswith('.py'):
                raw_url = github_info.get('raw_url')
                if raw_url:
                    try:
                        response = requests.get(raw_url, headers=headers, timeout=timeout)
                        if response.status_code == 200:
                            file_content = response.text
                            file_name = Path(path).name
                            if _is_model_workflow_file(file_name, path):
                                extracted_code.append({
                                    'path': path,
                                    'name': file_name,
                                    'content': file_content
                                })
                                print(f"  [GitHub] Extracted file: {file_name}")
                    except Exception as e:
                        print(f"  [GitHub] Failed to fetch file {path}: {e}")
        else:
            # 仓库主页或目录，递归提取Python文件
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}" if path else f"https://api.github.com/repos/{owner}/{repo}/contents"
            # print(api_url)
            try:
                # 递归获取所有Python文件（收集更多文件以便后续按优先级选择）
                python_files = _get_python_files_recursive(api_url, github_headers, timeout, max_collect=50)
                
                if not python_files:
                    print(f"  [GitHub] No Python files found matching criteria. Trying to collect all .py files...")
                    # 如果没找到文件，尝试更宽松的收集策略
                    python_files = _get_python_files_recursive(api_url, github_headers, timeout, max_collect=50, strict_filter=False)
                
                if not python_files:
                    print(f"  [GitHub] Still no files found. This might be a non-Python repository or files are in excluded directories.")
                    return None
                
                # 按优先级排序：优先文件名 > 路径深度
                def get_priority(file_item):
                    file_name = file_item.get('name', '').lower()
                    file_path = file_item.get('path', '')
                    path_parts = file_path.split('/')
                    depth = len(path_parts) - 1
                    
                    priority_names = {
                        'model.py', 'models.py', 'modeling.py',
                        'train.py', 'training.py', 'trainer.py',
                        'main.py', 'run.py', 'pipeline.py', 'workflow.py',
                        'app.py', 'server.py', 'api.py'
                    }
                    
                    if file_name in priority_names:
                        return (0, depth)  # 最高优先级
                    elif depth <= 1:
                        return (1, depth)  # 根目录文件
                    else:
                        return (2, depth)  # 其他文件
                
                python_files.sort(key=get_priority)
                print(f"  [GitHub] Found {len(python_files)} Python files, selecting top {max_files}")
                
                # 提取文件内容
                for item in python_files[:max_files]:
                    file_name = item.get('name', '')
                    file_path = item.get('path', '')
                    download_url = item.get('download_url') or item.get('url', '')
                    
                    try:
                        if download_url and download_url.startswith('http'):
                            file_response = requests.get(download_url, headers=headers, timeout=timeout)
                            if file_response.status_code == 200:
                                file_content = file_response.text
                                extracted_code.append({
                                    'path': file_path,
                                    'name': file_name,
                                    'content': file_content
                                })
                                print(f"  [GitHub] Extracted file: {file_name} ({file_path})")
                    except Exception as e:
                        print(f"  [GitHub] Failed to fetch file {file_name}: {e}")
                        continue
                            
            except Exception as e:
                print(f"  [GitHub] API request failed: {e}")
        
        if not extracted_code:
            print(f"  [GitHub] No model workflow code files extracted from {url}")
            return None
        
        # 保存提取的代码
        base_filename = sanitize_filename(title)
        filename = f"{base_filename}_code.txt"
        code_path = save_dir / filename
        
        # 组合所有代码文件内容
        combined_content = []
        combined_content.append(f"# GitHub Repository: {owner}/{repo}\n")
        combined_content.append(f"# URL: {url}\n")
        combined_content.append(f"# Extracted {len(extracted_code)} Python file(s) related to model building and workflow\n")
        combined_content.append("=" * 80 + "\n\n")
        
        for code_file in extracted_code:
            combined_content.append(f"# File: {code_file['name']}\n")
            combined_content.append(f"# Path: {code_file['path']}\n")
            combined_content.append("-" * 80 + "\n")
            combined_content.append(code_file['content'])
            combined_content.append("\n\n" + "=" * 80 + "\n\n")
        
        try:
            with open(code_path, 'w') as f:
                f.write(''.join(combined_content))
            print(f"  [GitHub] Saved {len(extracted_code)} Python files to: {code_path}")
            return code_path
        except Exception as e:
            print(f"  [GitHub] Failed to save code file: {e}")
            return None
            
    except Exception as e:
        print(f"  [GitHub] Error extracting code: {e}")
        return None


def fetch_urls_from_log(
    log_file: Path,
    dataset_name: str,
    timeout: int = 30,
    max_content_length: int = 10 * 1024 * 1024,  # 10MB
    github_token: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    从日志文件中读取所有URL，并爬取对应的网页内容。
    对于每个URL，首先检查是否有PDF可下载，如果有则下载PDF，否则解析HTML内容。
    
    Args:
        log_file: 日志文件路径
        dataset_name: 数据集名称
        timeout: 请求超时时间（秒）
        max_content_length: 最大内容长度（字节），超过此长度将截断
        github_token: GitHub Personal Access Token（可选，但强烈建议提供以提高速率限制）
        
    Returns:
        字典，格式为 {target_description: all_results}，其中 all_results 中的每个结果
        包含原有的所有字段（title, url, snippet等），并新增了 "path" 字段（PDF或TXT文件路径）。
    """
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    print(f"\n[Fetch URLs] Reading log file: {log_file}")
    
    # 创建PDF保存目录（每个dataset一个目录）
    # log_file现在在 dataset 子目录中，所以 log_file.parent 就是 dataset_dir
    dataset_dir = log_file.parent
    pdf_dir = dataset_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Fetch URLs] PDF save directory: {pdf_dir}")
    
    # 存储每个target_description对应的all_results
    target_to_results: Dict[str, List[Dict[str, Any]]] = {}
    
    # 读取JSONL文件
    with open(log_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            log_record = json.loads(line)
            # log_record格式: {target_description: result}
            for target_desc, result in log_record.items():
                # 从 all_results 中提取结果
                if "all_results" in result:
                    all_results = result["all_results"]
                    # 深拷贝结果，避免修改原始数据
                    all_results_copy = [res.copy() for res in all_results]
                    # 如果target_description已存在，合并结果而不是覆盖
                    if target_desc in target_to_results:
                        target_to_results[target_desc].extend(all_results_copy)
                    else:
                        target_to_results[target_desc] = all_results_copy
    
    print(f"[Fetch URLs] Processing {len(target_to_results)} target descriptions")
    
    # 爬取每个URL的内容
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    total_results = sum(len(results) for results in target_to_results.values())
    result_count = 0
    success_count = 0
    pdf_count = 0
    txt_count = 0
    github_code_count = 0
    # 用于跟踪已使用的文件名，避免重复
    used_filenames = set()
    
    # 遍历每个target_description的all_results
    for target_desc, all_results in target_to_results.items():
        print(f"\n[Fetch URLs] Processing target: {target_desc}...")
        
        for res in all_results:
            url = res.get("url", "").strip()
            if not url:
                res["path"] = ""
                continue
            
            # 规范化 arxiv URL
            url = normalize_arxiv_url(url)
            
            result_count += 1
            print(f"[Fetch URLs] [{result_count}/{total_results}] Fetching: {url}")
            
            fetch_error = None
            pdf_path = None
            txt_path = None
            code_path = None
            title = res.get('title', '').strip() or 'untitled'
            
            # 检查是否是 GitHub URL 且是 Strategy C，如果是则使用特殊处理
            is_strategy_c = 'Strategy C' in target_desc or 'Code Implementation' in target_desc
            if is_github_url(url) and is_strategy_c:
                print(f"  [GitHub] Detected GitHub URL in Strategy C, extracting code...")
                code_path = extract_github_code(url, pdf_dir, title, headers, timeout, github_token=github_token)
                if code_path:
                    res['path'] = str(code_path)
                    github_code_count += 1
                    success_count += 1
                else:
                    res['path'] = ""
                    fetch_error = "Failed to extract GitHub code"
                continue
            
            try:
                # continue
                # 发送HTTP请求
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=True,
                )
                
                if response.status_code != 200:
                    fetch_error = f"HTTP {response.status_code}"
                    print(f"  [Error] HTTP {response.status_code}")
                else:
                    # 检查内容长度
                    content_length = len(response.content)
                    if content_length > max_content_length:
                        print(f"  [Warning] Content too large ({content_length} bytes), truncating to {max_content_length} bytes")
                        response_content = response.content[:max_content_length]
                    else:
                        response_content = response.content
                    
                    # 检查Content-Type，如果是PDF直接保存
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'pdf' in content_type or url.lower().endswith('.pdf'):
                        # 直接是PDF文件，使用title命名
                        base_filename = sanitize_filename(title)
                        filename = f"{base_filename}.pdf"
                        
                        # 确保文件名唯一
                        counter = 1
                        while filename in used_filenames:
                            filename = f"{base_filename}_{counter}.pdf"
                            counter += 1
                        used_filenames.add(filename)
                        
                        pdf_path = pdf_dir / filename
                        
                        if download_pdf(url, pdf_path, headers, timeout):
                            print(f"  [PDF] Downloaded: {pdf_path}")
                            pdf_count += 1
                            success_count += 1
                        else:
                            fetch_error = "Failed to download PDF"
                    else:
                        # 尝试解析HTML并查找PDF链接
                        try:
                            # 检测编码
                            if response.encoding:
                                html_content = response_content.decode(response.encoding, errors='ignore')
                            else:
                                # 尝试常见编码
                                for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                                    try:
                                        html_content = response_content.decode(encoding)
                                        break
                                    except UnicodeDecodeError:
                                        continue
                                else:
                                    html_content = response_content.decode('utf-8', errors='ignore')
                            
                            # 使用BeautifulSoup解析HTML
                            soup = BeautifulSoup(html_content, 'html.parser')
                            
                            # 查找PDF链接
                            pdf_links = find_pdf_links(soup, url)
                            
                            if pdf_links:
                                # 尝试下载第一个PDF链接
                                pdf_downloaded = False
                                for pdf_link in pdf_links[:3]:  # 最多尝试3个PDF链接
                                    try:
                                        # 使用title命名PDF文件
                                        base_filename = sanitize_filename(title)
                                        filename = f"{base_filename}.pdf"
                                        
                                        # 确保文件名唯一
                                        counter = 1
                                        while filename in used_filenames:
                                            filename = f"{base_filename}_{counter}.pdf"
                                            counter += 1
                                        used_filenames.add(filename)
                                        
                                        pdf_path = pdf_dir / filename
                                        
                                        if download_pdf(pdf_link, pdf_path, headers, timeout):
                                            print(f"  [PDF] Found and downloaded: {pdf_path}")
                                            pdf_downloaded = True
                                            pdf_count += 1
                                            success_count += 1
                                            break
                                    except Exception as e:
                                        print(f"    [PDF Link Error] {pdf_link}: {e}")
                                        continue
                                
                                if not pdf_downloaded:
                                    # PDF下载失败，继续解析HTML内容并保存为txt
                                    print(f"  [Info] PDF links found but download failed, parsing HTML content")
                                    # 移除script和style标签
                                    for script in soup(["script", "style"]):
                                        script.decompose()
                                    
                                    # 提取文本
                                    text = soup.get_text(separator='\n', strip=True)
                                    
                                    # 清理多余的空白行
                                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                                    content = '\n'.join(lines)
                                    
                                    # 保存为txt文件
                                    base_filename = sanitize_filename(title)
                                    filename = f"{base_filename}.txt"
                                    counter = 1
                                    while filename in used_filenames:
                                        filename = f"{base_filename}_{counter}.txt"
                                        counter += 1
                                    used_filenames.add(filename)
                                    txt_path = pdf_dir / filename
                                    
                                    try:
                                        with open(txt_path, 'w') as f:
                                            f.write(content)
                                        print(f"  [TXT] Saved content to: {txt_path}")
                                        txt_count += 1
                                    except Exception as e:
                                        print(f"  [Warning] Failed to save txt file: {e}")
                                    
                                    print(f"  [Success] Fetched {len(content)} characters")
                                    success_count += 1
                            else:
                                # 没有找到PDF链接，解析HTML内容并保存为txt
                                # 移除script和style标签
                                for script in soup(["script", "style"]):
                                    script.decompose()
                                
                                # 提取文本
                                text = soup.get_text(separator='\n', strip=True)
                                
                                # 清理多余的空白行
                                lines = [line.strip() for line in text.split('\n') if line.strip()]
                                content = '\n'.join(lines)
                                
                                # 保存为txt文件
                                base_filename = sanitize_filename(title)
                                filename = f"{base_filename}.txt"
                                counter = 1
                                while filename in used_filenames:
                                    filename = f"{base_filename}_{counter}.txt"
                                    counter += 1
                                used_filenames.add(filename)
                                txt_path = pdf_dir / filename
                                
                                try:
                                    with open(txt_path, 'w') as f:
                                        f.write(content)
                                    print(f"  [TXT] Saved content to: {txt_path}")
                                    txt_count += 1
                                except Exception as e:
                                    print(f"  [Warning] Failed to save txt file: {e}")
                                
                                print(f"  [Success] Fetched {len(content)} characters")
                                success_count += 1
                                
                        except Exception as e:
                            # 如果HTML解析失败，尝试直接解码为文本并保存为txt
                            print(f"  [Warning] HTML parsing failed, using raw text: {e}")
                            if response.encoding:
                                content = response_content.decode(response.encoding, errors='ignore')
                            else:
                                content = response_content.decode('utf-8', errors='ignore')
                            
                            # 保存为txt文件
                            base_filename = sanitize_filename(title)
                            filename = f"{base_filename}.txt"
                            counter = 1
                            while filename in used_filenames:
                                filename = f"{base_filename}_{counter}.txt"
                                counter += 1
                            used_filenames.add(filename)
                            txt_path = pdf_dir / filename
                            
                            try:
                                with open(txt_path, 'w') as f:
                                    f.write(content)
                                print(f"  [TXT] Saved content to: {txt_path}")
                                txt_count += 1
                            except Exception as e2:
                                print(f"  [Warning] Failed to save txt file: {e2}")
                            
                            success_count += 1
                    
            except requests.exceptions.Timeout:
                fetch_error = "Request timeout"
                print(f"  [Error] Request timeout")
            except requests.exceptions.RequestException as e:
                fetch_error = str(e)
                print(f"  [Error] Request failed: {e}")
            except Exception as e:
                fetch_error = str(e)
                print(f"  [Error] Unexpected error: {e}")
            
            # 将内容添加到结果中
            res['path'] = ''
            if pdf_path:
                res["path"] = str(pdf_path)
            elif txt_path:
                res["path"] = str(txt_path)
            elif code_path:
                res["path"] = str(code_path)
    
    print(f"\n[Fetch URLs] Completed. Successfully fetched {success_count}/{total_results} URLs")
    print(f"[Fetch URLs] PDF files downloaded: {pdf_count}")
    print(f"[Fetch URLs] TXT files saved: {txt_count}")
    print(f"[Fetch URLs] GitHub code files extracted: {github_code_count}")
    print(f"[Fetch URLs] GitHub code files extracted: {github_code_count}")
    
    return target_to_results


def read_file_content(file_path: Path, max_length: int = 200000) -> Optional[str]:
    """
    读取文件内容（PDF或TXT）。
    
    Args:
        file_path: 文件路径
        max_length: 最大读取长度（字符数）
        
    Returns:
        文件内容字符串，如果读取失败返回None
    """
    if not file_path.exists():
        print(f"  [Warning] File not found: {file_path}")
        return None
    
    try:
        if file_path.suffix.lower() == '.pdf':
            # 读取PDF内容
            if PDF_AVAILABLE:
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text_content = ""
                        for page in pdf_reader.pages:
                            text_content += page.extract_text() + "\n"
                        return text_content[:max_length]
                except Exception as e:
                    print(f"    [Error] Failed to read PDF with PyPDF2: {e}")
                    try:
                        import pdfplumber
                        with pdfplumber.open(file_path) as pdf:
                            text_content = ""
                            for page in pdf.pages:
                                text_content += page.extract_text() + "\n"
                            return text_content[:max_length]
                    except Exception as e2:
                        print(f"    [Error] Failed to read PDF with pdfplumber: {e2}")
                        return None
            else:
                print(f"    [Warning] PDF reading libraries not available")
                return None
        elif file_path.suffix.lower() == '.txt':
            # 读取TXT内容
            with open(file_path, 'r') as f:
                content = f.read()
                return content[:max_length]
        else:
            print(f"  [Warning] Unsupported file type: {file_path.suffix}")
            return None
    except Exception as e:
        print(f"    [Error] Failed to read file: {e}")
        return None

