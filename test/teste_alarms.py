import sys
import os

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from bibmon._alarms import detecOutlier


def test_detect_drift():
    data = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7])  # Valores crescentes (drift)
    lim = 1.0

    # Função detectará um drift quando houver N aumentos consecutivos
    expected_output = 1  # Alarme ativado

    result = detecOutlier(data, lim, method="drift")

    assert result == expected_output, f"Saída inesperada: {result}"

def test_nelson_rule_1():
    data = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])  # 8 valores acima do limiar
    lim = 1.0

    expected_output = 1  # Alarme ativado

    result = detecOutlier(data, lim, method="nelson_rule_1")

    assert result == expected_output, f"Saída inesperada: {result}"