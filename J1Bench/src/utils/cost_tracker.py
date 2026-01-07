"""
Cost Tracker for tracking API costs per case/sample.

Each case should have its own CostTracker instance to avoid data corruption in async environments.
"""


class CostTracker:
    """Tracks token usage and costs for a single case/sample"""
    
    def __init__(self):
        self.total_cost = 0.0
    
    def add_cost(self, amount: float) -> None:
        """
        Add cost to this tracker.
        
        Args:
            amount: Cost amount to add
        """
        if amount is None:
            return
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return
        if amount <= 0:
            return
        self.total_cost += amount
    
    def get_total_cost(self) -> float:
        """Get total cost for this tracker"""
        return self.total_cost
    
    def reset(self) -> None:
        """Reset the cost tracker"""
        self.total_cost = 0.0


# 为了向后兼容，保留全局函数（但建议使用 CostTracker 类）
import threading
from typing import Optional, Dict

# 全局总成本（所有任务的总和）
_total_cost = 0.0
# 每个任务的成本跟踪（键：任务ID，值：成本）
_task_costs: Dict[str, float] = {}
# 当前任务ID（使用线程本地存储）
_local = threading.local()
_lock = threading.Lock()


def set_task_id(task_id: Optional[str] = None) -> None:
    """
    设置当前线程的任务ID（已废弃，建议使用 CostTracker 类）
    
    Args:
        task_id: 任务ID，如果为None则使用线程ID
    """
    if task_id is None:
        task_id = str(threading.get_ident())
    _local.task_id = task_id
    
    # 初始化该任务的成本为0（如果还没有）
    with _lock:
        if task_id not in _task_costs:
            _task_costs[task_id] = 0.0


def get_task_id() -> Optional[str]:
    """获取当前线程的任务ID（已废弃，建议使用 CostTracker 类）"""
    return getattr(_local, 'task_id', None)


def add_cost(amount: float, task_id: Optional[str] = None) -> None:
    """
    Accumulate monetary cost in a thread-safe way (已废弃，建议使用 CostTracker 类).
    
    Args:
        amount: 要添加的成本金额
        task_id: 任务ID，如果为None则使用当前线程的任务ID
    """
    if amount is None:
        return
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return
    if amount <= 0:
        return
    
    # 如果没有指定task_id，尝试使用当前线程的任务ID
    if task_id is None:
        task_id = get_task_id()
    
    with _lock:
        global _total_cost
        _total_cost += amount
        
        # 如果指定了任务ID，也累加到该任务的成本中
        if task_id is not None:
            if task_id not in _task_costs:
                _task_costs[task_id] = 0.0
            _task_costs[task_id] += amount


def get_total_cost() -> float:
    """获取全局总成本（所有任务的总和）（已废弃，建议使用 CostTracker 类）"""
    with _lock:
        return _total_cost


def get_task_cost(task_id: Optional[str] = None) -> float:
    """
    获取指定任务的成本（已废弃，建议使用 CostTracker 类）
    
    Args:
        task_id: 任务ID，如果为None则使用当前线程的任务ID
        
    Returns:
        该任务的成本，如果任务不存在则返回0.0
    """
    if task_id is None:
        task_id = get_task_id()
    
    if task_id is None:
        return 0.0
    
    with _lock:
        return _task_costs.get(task_id, 0.0)


def reset_total_cost() -> None:
    """
    重置全局总成本（谨慎使用，会影响所有任务）（已废弃，建议使用 CostTracker 类）
    """
    with _lock:
        global _total_cost
        _total_cost = 0.0


def reset_task_cost(task_id: Optional[str] = None) -> None:
    """
    重置指定任务的成本（已废弃，建议使用 CostTracker 类）
    
    Args:
        task_id: 任务ID，如果为None则使用当前线程的任务ID
    """
    if task_id is None:
        task_id = get_task_id()
    
    if task_id is None:
        return
    
    with _lock:
        if task_id in _task_costs:
            # 从全局总成本中减去该任务的成本
            global _total_cost
            _total_cost -= _task_costs[task_id]
            # 重置该任务的成本
            _task_costs[task_id] = 0.0


def get_all_task_costs() -> Dict[str, float]:
    """
    获取所有任务的成本字典（已废弃，建议使用 CostTracker 类）
    
    Returns:
        任务ID到成本的映射字典
    """
    with _lock:
        return _task_costs.copy()
