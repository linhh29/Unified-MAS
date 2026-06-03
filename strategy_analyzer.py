"""
Strategy分析模块
负责按Strategy分类结果，读取文件内容，并使用LLM进行分析
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any

from Unified_MAS.content_fetcher import read_file_content
from Unified_MAS.prompts import get_strategy_analysis_prompt


def clean_json_response(text: str) -> str:
    """
    清理LLM返回的JSON字符串，移除markdown代码块、注释等。
    
    Args:
        text: 原始JSON字符串
        
    Returns:
        清理后的JSON字符串
    """
    if not text:
        return "{}"
    
    # 移除markdown代码块
    if "```json" in text:
        json_start = text.find("```json") + 7
        json_end = text.find("```", json_start)
        if json_end != -1:
            text = text[json_start:json_end].strip()
    elif "```" in text:
        json_start = text.find("```") + 3
        json_end = text.find("```", json_start)
        if json_end != -1:
            text = text[json_start:json_end].strip()
            if text.startswith("json"):
                text = text[4:].strip()
    
    # 移除行注释和尾随逗号
    lines = []
    for line in text.split('\n'):
        # 移除单行注释
        if '//' in line:
            comment_pos = line.find('//')
            # 检查是否在字符串中
            in_string = False
            quote_char = None
            for i, char in enumerate(line):
                if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                elif i == comment_pos and not in_string:
                    line = line[:i].rstrip()
                    break
        lines.append(line)
    
    text = '\n'.join(lines)
    
    # 移除尾随逗号（在 } 或 ] 之前）
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    return text.strip()


def fix_unicode_surrogates(obj):
    """
    递归修复字典/列表中的Unicode代理对问题。
    
    Args:
        obj: 要修复的对象（dict, list, str等）
        
    Returns:
        修复后的对象
    """
    if isinstance(obj, dict):
        return {k: fix_unicode_surrogates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_unicode_surrogates(item) for item in obj]
    elif isinstance(obj, str):
        # 修复Unicode代理对
        try:
            # 尝试编码为UTF-8，如果失败则修复
            obj.encode('utf-8')
            return obj
        except UnicodeEncodeError:
            # 替换无效的代理对字符
            return obj.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    else:
        return obj


def classify_results_by_strategy(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    按Strategy分类结果。
    
    Args:
        all_results: {target_description: all_results}格式的字典
        
    Returns:
        {strategy_name: results}格式的字典
    """
    strategy_to_results: Dict[str, List[Dict[str, Any]]] = {}
    
    for target_desc, results in all_results.items():
        if 'Strategy A' in target_desc or 'Background Knowledge' in target_desc:
            strategy_name = "Strategy A - Background Knowledge"
        elif 'Strategy B' in target_desc or 'System Architecture' in target_desc:
            strategy_name = "Strategy B - High-quality Academic Papers about System Architecture (Workflow & Design)"
        elif 'Strategy C' in target_desc or 'Code Implementation' in target_desc:
            strategy_name = "Strategy C - Technical Code Implementation"
        elif 'Strategy D' in target_desc or 'Evaluation' in target_desc:
            strategy_name = "Strategy D - Evaluation & Metrics"
        else:
            strategy_name = "Unknown Strategy"
        
        if strategy_name not in strategy_to_results:
            strategy_to_results[strategy_name] = []
        strategy_to_results[strategy_name].extend(results)
    
    return strategy_to_results


def analyze_strategy_files(
    strategy_name: str,
    results: List[Dict[str, Any]],
    task_thinking: str,
    llm_client,
    intermediate_dir: Path,
) -> Dict[str, Any]:
    """
    分析某个Strategy下的所有文件。
    
    Args:
        strategy_name: Strategy名称
        results: 该Strategy下的所有结果
        task_thinking: 任务思考描述
        llm_client: LLM客户端
        intermediate_dir: 中间结果目录
        
    Returns:
        分析结果字典
    """
    print(f"\n{'='*80}")
    print(f"[Analysis] Processing {strategy_name}")
    print(f"[Analysis] Found {len(results)} results")
    print(f"{'='*80}")
    
    # 收集所有文件内容
    file_contents = []
    for res in results:
        file_path = res.get("path", "").strip()
        if not file_path:
            continue
        
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"  [Warning] File not found: {file_path}")
            continue
        
        title = res.get("title", "Unknown")
        print(f"  [Reading] {title}: {file_path.name}")
        
        content = read_file_content(file_path, max_length=200000)
        if content is not None:
            # 清理无法编码的字符（如表情符号等）
            try:
                # 尝试编码为UTF-8，如果失败则使用ignore模式清理
                content.encode('utf-8')
            except UnicodeEncodeError:
                print(f"  [Warning] File '{title}' contains invalid Unicode characters, cleaning...")
                content = content.encode('utf-8', 'ignore').decode('utf-8')
            
            file_contents.append({
                "title": title,
                "url": res.get("url", ""),
                "content": content
            })
    
    if not file_contents:
        print(f"  [Warning] No valid files found for {strategy_name}")
        return {"error": "No valid files found for analysis."}
    
    # 使用LLM分析这些文件
    print(f"  [Analysis] Analyzing {len(file_contents)} files with LLM...")
    
    # 构建文件内容摘要
    files_summary = []
    for i, file_info in enumerate(file_contents, 1):
        # 清理内容预览中的无效字符
        content_preview = file_info['content'][:10000]
        try:
            content_preview.encode('utf-8')
        except UnicodeEncodeError:
            content_preview = content_preview.encode('utf-8', 'ignore').decode('utf-8')
        
        summary_item = (
            f"Document {i}: {file_info['title']}\n"
            f"URL: {file_info['url']}\n"
            f"Content Preview (first 10000 chars):\n{content_preview}...\n"
        )
        
        # 清理摘要项中的无效字符
        try:
            summary_item.encode('utf-8')
        except UnicodeEncodeError:
            summary_item = summary_item.encode('utf-8', 'ignore').decode('utf-8')
        
        files_summary.append(summary_item)
    
    # 最后检查整个摘要列表（在发送给LLM前）
    try:
        # 尝试将摘要列表转换为字符串并编码
        summary_test = '\n'.join(files_summary)
        summary_test.encode('utf-8')
    except UnicodeEncodeError:
        print(f"  [Warning] Combined summary has encoding issues, cleaning all items...")
        files_summary = [item.encode('utf-8', 'ignore').decode('utf-8') for item in files_summary]
    
    # 根据Strategy生成相应的prompt
    analysis_system, analysis_user = get_strategy_analysis_prompt(strategy_name, task_thinking, files_summary)
    
    analysis_messages = [
        {"role": "system", "content": analysis_system},
        {"role": "user", "content": analysis_user},
    ]
    
    try:
        analysis_result_str = llm_client.chat(analysis_messages, response_format='json_object')
        
        # 清理JSON响应
        cleaned_json = clean_json_response(analysis_result_str)
        
        # 尝试解析JSON
        try:
            analysis_result = json.loads(cleaned_json)
        except json.JSONDecodeError as json_err:
            print(f"  [Warning] JSON parsing failed, attempting to fix...")
            print(f"  [Debug] First 500 chars of response: {cleaned_json[:500]}")
            
            # 尝试更激进的修复
            # 移除所有控制字符（除了换行和制表符）
            cleaned_json = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned_json)
            
            # 再次尝试解析
            try:
                analysis_result = json.loads(cleaned_json)
            except json.JSONDecodeError:
                # 如果还是失败，尝试提取第一个完整的JSON对象
                try:
                    # 查找第一个 { 和最后一个 }
                    first_brace = cleaned_json.find('{')
                    last_brace = cleaned_json.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_str = cleaned_json[first_brace:last_brace+1]
                        analysis_result = json.loads(json_str)
                    else:
                        raise json_err
                except:
                    print(f"  [Error] Could not parse JSON after cleaning: {json_err}")
                    print(f"  [Debug] Cleaned JSON (first 1000 chars): {cleaned_json[:1000]}")
                    raise json_err
        
        # 修复Unicode代理对问题
        analysis_result = fix_unicode_surrogates(analysis_result)
        
        print(f"  [Success] Analysis completed for {strategy_name}")
        return analysis_result
    except Exception as e:
        print(f"  [Error] Failed to analyze {strategy_name}: {e}")
        import traceback
        print(f"  [Traceback] {traceback.format_exc()}")
        return {"error": str(e)}


def analyze_all_strategies(
    all_results: Dict[str, List[Dict[str, Any]]],
    task_thinking: str,
    llm_client,
    dataset: str,
    intermediate_dir: Path,
) -> Dict[str, Any]:
    """
    分析所有Strategy。
    如果结果文件已存在，只重新分析有error的strategy。
    
    Args:
        all_results: {target_description: all_results}格式的字典
        task_thinking: 任务思考描述
        llm_client: LLM客户端
        dataset: 数据集名称
        intermediate_dir: 中间结果目录
        
    Returns:
        {strategy_name: analysis_result}格式的字典
    """
    print(f"\n[Analysis] Total targets: {len(all_results)}")
    
    # 按Strategy分类结果
    strategy_to_results = classify_results_by_strategy(all_results)
    
    print(f"\n[Analysis] Strategies found: {list(strategy_to_results.keys())}")
    
    
    # 检查是否存在已有的分析结果文件
    analysis_output_file = intermediate_dir / "strategy_analysis.json"
    print(f"analysis_output_file: {analysis_output_file}")
    strategy_summaries = {}
    
    if analysis_output_file.exists():
        print(f"\n[Analysis] Found existing analysis file: {analysis_output_file}")
        try:
            with open(analysis_output_file, 'r') as f:
                strategy_summaries = json.load(f)
            print(f"[Analysis] Loaded {len(strategy_summaries)} existing strategy results")
            
            # 检查哪些strategy有error或缺失
            strategies_to_rerun = []
            for strategy_name, results in strategy_to_results.items():
                if strategy_name not in strategy_summaries:
                    print(f"  [Info] Strategy '{strategy_name}' not found in existing results, will analyze")
                    strategies_to_rerun.append(strategy_name)
                elif isinstance(strategy_summaries.get(strategy_name), dict) and "error" in strategy_summaries[strategy_name]:
                    print(f"  [Info] Strategy '{strategy_name}' has error: {strategy_summaries[strategy_name].get('error', 'Unknown error')}")
                    print(f"  [Info] Will re-analyze this strategy")
                    strategies_to_rerun.append(strategy_name)
                else:
                    print(f"  [Info] Strategy '{strategy_name}' already has valid results, skipping")
            
            if not strategies_to_rerun:
                print(f"\n[Analysis] All strategies have valid results. No re-analysis needed.")
                return strategy_summaries
            else:
                print(f"\n[Analysis] Will re-analyze {len(strategies_to_rerun)} strategy(ies): {strategies_to_rerun}")
        except Exception as e:
            print(f"[Warning] Failed to load existing analysis file: {e}")
            print(f"[Info] Will perform full analysis from scratch")
            strategy_summaries = {}
            strategies_to_rerun = list(strategy_to_results.keys())
    else:
        print(f"\n[Analysis] No existing analysis file found. Will perform full analysis.")
        strategies_to_rerun = list(strategy_to_results.keys())
    
    # 为需要重新分析的Strategy生成分析总结
    for strategy_name, results in strategy_to_results.items():
        if strategy_name in strategies_to_rerun:
            print(f"\n[Analysis] Analyzing {strategy_name}...")
            analysis_result = analyze_strategy_files(
                strategy_name, results, task_thinking, llm_client, intermediate_dir
            )
            strategy_summaries[strategy_name] = analysis_result
        else:
            print(f"\n[Analysis] Skipping {strategy_name} (already has valid results)")
    
    # 保存分析结果（合并已有结果和新结果）
    analysis_output_file = intermediate_dir / "strategy_analysis.json"
    try:
        # 修复所有Unicode代理对问题
        fixed_summaries = fix_unicode_surrogates(strategy_summaries)
        
        with open(analysis_output_file, 'w') as f:
            json.dump(fixed_summaries, f, ensure_ascii=False, indent=2)
        print(f"\n[Analysis] Results saved to: {analysis_output_file}")
    except UnicodeEncodeError as e:
        print(f"\n[Error] Failed to save analysis results due to encoding error: {e}")
        # 尝试使用更宽松的编码方式
        try:
            fixed_summaries = fix_unicode_surrogates(strategy_summaries)
            with open(analysis_output_file, 'w') as f:
                json.dump(fixed_summaries, f, ensure_ascii=False, indent=2)
            print(f"[Analysis] Results saved with encoding fixes to: {analysis_output_file}")
        except Exception as e2:
            print(f"[Error] Could not save analysis results: {e2}")
    except Exception as e:
        print(f"\n[Error] Failed to save analysis results: {e}")
    
    # 打印摘要
    print(f"\n{'='*80}")
    print("STRATEGY ANALYSIS SUMMARIES")
    print(f"{'='*80}")
    for strategy_name, summary in strategy_summaries.items():
        print(f"\n{strategy_name}:")
        if isinstance(summary, dict) and "error" not in summary:
            # 根据不同的Strategy打印不同的字段
            if "aspects_covered" in summary:
                print(f"  Aspects Covered: {summary.get('aspects_covered', [])}")
            elif "architectural_patterns" in summary:
                print(f"  Architectural Patterns: {summary.get('architectural_patterns', [])}")
            elif "implementation_approaches" in summary:
                print(f"  Implementation Approaches: {summary.get('implementation_approaches', [])}")
            elif "evaluation_metrics" in summary:
                print(f"  Evaluation Metrics: {summary.get('evaluation_metrics', [])}")
            print(f"  Summary: {summary.get('summary', '')[:500]}...")
        else:
            print(f"  {summary}")
    
    return strategy_summaries

