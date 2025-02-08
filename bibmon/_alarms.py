import numpy as np

def detecOutlier(data, lim, count=False, count_limit=1, method="basic"):
    """
    Detecta outliers com base em diferentes métodos (básico, drift, regras de Nelson).
    
    Parâmetros:
    - data: array-like, dados de entrada.
    - lim: float, limite para detectar outliers.
    - count: bool, se True, conta os outliers.
    - count_limit: int, número máximo de outliers permitidos antes de acionar o alarme.
    - method: str, define o método a ser utilizado para detecção ("basic", "drift", "nelson_rule_1").
    
    Retorna:
    - alarm: ndarray ou int, indicando os outliers detectados.
    """

    # Tratar valores NaN
    if np.isnan(data).any():
        data = np.nan_to_num(data)

    # Método básico: Detectar valores acima do limite
    if method == "basic":
        return _basic_outlier_detection(data, lim, count, count_limit)

    # Método drift: Detectar aumento contínuo
    elif method == "drift":
        return _drift_detection(data)

    # Método Nelson Rule 1: Detectar 8 pontos consecutivos acima do limite
    elif method == "nelson_rule_1":
        return _nelson_rule_1(data, lim)

    return 0  # Se nenhum método for aplicado

def _basic_outlier_detection(data, lim, count=False, count_limit=1):
    """Detecta outliers básicos, baseado no limite fornecido."""
    if count is False:
        alarm = np.copy(data)
        alarm = np.where(alarm <= lim, 0, alarm)
        alarm = np.where(alarm > lim, 1, alarm)
        return alarm
    else:
        local_count = np.count_nonzero(data > lim)
        return 1 if local_count > count_limit else 0

def _drift_detection(data):
    """Detecta aumento contínuo nos dados (drift)."""
    diffs = np.diff(data)
    if np.all(diffs > 0):  # Todos os valores estão aumentando
        return 1  # Alarme de drift ativado
    return 0

def _nelson_rule_1(data, lim):
    """Aplica a regra de Nelson: 8 valores consecutivos acima do limite."""
    above_lim = data > lim
    count_above = np.convolve(above_lim, np.ones(8, dtype=int), mode="valid")
    if np.any(count_above == 8):
        return 1  # Alarme ativado
    return 0
