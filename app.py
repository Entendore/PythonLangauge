"""
Permutation → Python Code Validator — 10K Sample Edition
Streaming statistics, live animated visuals, 10,000 sample capacity,
QAbstractTableModel for high-performance tables, export, presets, dark mode.
"""

import sys
import math
import itertools
import csv
import time
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Tuple, Optional, List, Dict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QComboBox,
    QProgressBar, QGroupBox, QTableView, QSplitter, QHeaderView,
    QMessageBox, QTabWidget, QFrame, QFileDialog, QShortcut
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QAbstractTableModel, QModelIndex, QSize
)
from PySide6.QtGui import QAction, QColor, QPalette
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  DATACLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class PermStats:
    valid_count: int = 0
    invalid_count: int = 0
    total: int = 0
    processed: int = 0
    error_types: Counter = field(default_factory=Counter)
    error_samples: list = field(default_factory=list)
    valid_samples: list = field(default_factory=list)
    invalid_samples: list = field(default_factory=list)
    by_length_valid: Counter = field(default_factory=Counter)
    by_length_invalid: Counter = field(default_factory=Counter)
    char_pos_valid: defaultdict = field(default_factory=lambda: defaultdict(Counter))
    char_pos_invalid: defaultdict = field(default_factory=lambda: defaultdict(Counter))
    adjacency_valid: Counter = field(default_factory=Counter)
    adjacency_invalid: Counter = field(default_factory=Counter)
    first_char_valid: Counter = field(default_factory=Counter)
    first_char_invalid: Counter = field(default_factory=Counter)
    characters: str = ""
    max_length: int = 0

    def snapshot(self) -> 'PermStats':
        s = PermStats()
        s.valid_count = self.valid_count
        s.invalid_count = self.invalid_count
        s.total = self.total
        s.processed = self.processed
        s.error_types = Counter(self.error_types)
        s.error_samples = list(self.error_samples)
        s.valid_samples = list(self.valid_samples)
        s.invalid_samples = list(self.invalid_samples)
        s.by_length_valid = Counter(self.by_length_valid)
        s.by_length_invalid = Counter(self.by_length_invalid)
        s.char_pos_valid = defaultdict(Counter, {k: Counter(v) for k, v in self.char_pos_valid.items()})
        s.char_pos_invalid = defaultdict(Counter, {k: Counter(v) for k, v in self.char_pos_invalid.items()})
        s.adjacency_valid = Counter(self.adjacency_valid)
        s.adjacency_invalid = Counter(self.adjacency_invalid)
        s.first_char_valid = Counter(self.first_char_valid)
        s.first_char_invalid = Counter(self.first_char_invalid)
        s.characters = self.characters
        s.max_length = self.max_length
        return s


# ═══════════════════════════════════════════════════════════════
#  HIGH-PERFORMANCE TABLE MODEL (handles 10,000+ rows)
# ═══════════════════════════════════════════════════════════════

class SampleModel(QAbstractTableModel):
    """Model that only queries data for visible rows — handles 10K+ smoothly."""

    COLUMNS = ["#", "Permutation", "Len"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: List[str] = []
        self._filtered: List[str] = []
        self._filter_text: str = ""

    def rowCount(self, parent=QModelIndex()):
        return len(self._filtered)

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLUMNS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        if row < 0 or row >= len(self._filtered):
            return None
        if role == Qt.DisplayRole:
            perm = self._filtered[row]
            col = index.column()
            if col == 0:
                return str(row + 1)
            elif col == 1:
                return perm
            elif col == 2:
                return str(len(perm))
        elif role == Qt.TextAlignmentRole:
            col = index.column()
            if col == 1:
                return int(Qt.AlignLeft | Qt.AlignVCenter)
            return int(Qt.AlignCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.COLUMNS[section]
            elif orientation == Qt.Vertical:
                return str(section + 1)
        return None

    def set_data(self, samples: list):
        self.beginResetModel()
        self._data = list(samples)
        self._apply_filter()
        self.endResetModel()

    def set_filter(self, text: str):
        self._filter_text = text.lower().strip()
        self.beginResetModel()
        self._apply_filter()
        self.endResetModel()

    def _apply_filter(self):
        if not self._filter_text:
            self._filtered = self._data
        else:
            self._filtered = [s for s in self._data if self._filter_text in s.lower()]

    def get_perm(self, row: int) -> Optional[str]:
        if 0 <= row < len(self._filtered):
            return self._filtered[row]
        return None

    def total_count(self) -> int:
        return len(self._data)

    def filtered_count(self) -> int:
        return len(self._filtered)


# ═══════════════════════════════════════════════════════════════
#  WORKER — 10,000 sample capacity
# ═══════════════════════════════════════════════════════════════

class PermutationWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    partial = Signal(PermStats)
    finished = Signal(PermStats)

    SAMPLE_SIZE = 10_000      # ← increased to 10,000
    ERROR_SAMPLE_SIZE = 200   # ← increased error samples too

    def __init__(self, characters: str, max_length: int):
        super().__init__()
        self.characters = characters
        self.max_length = max_length
        self._cancel = False
        self._pause = False

    def cancel(self):
        self._cancel = True

    def run(self):
        chars = self.characters
        n = min(self.max_length, len(chars))
        total = sum(math.perm(len(chars), l) for l in range(1, n + 1))

        stats = PermStats(total=total, characters=chars, max_length=n)
        self.status.emit(f"Streaming {total:,} permutations...")
        count = 0
        emit_interval = max(1, total // 120)
        start_time = time.time()

        for length in range(1, n + 1):
            for perm_tuple in itertools.permutations(chars, length):
                if self._cancel:
                    self.status.emit("Cancelled.")
                    return

                while self._pause and not self._cancel:
                    time.sleep(0.05)

                perm_str = ''.join(perm_tuple)
                count += 1
                stats.processed = count
                is_valid, error_cat, error_msg = self._check_valid_python(perm_str)

                if is_valid:
                    stats.valid_count += 1
                    stats.by_length_valid[length] += 1
                    stats.first_char_valid[perm_str[0]] += 1
                    for pos, ch in enumerate(perm_str):
                        stats.char_pos_valid[pos][ch] += 1
                    for j in range(len(perm_str) - 1):
                        stats.adjacency_valid[(perm_str[j], perm_str[j + 1])] += 1
                    if len(stats.valid_samples) < self.SAMPLE_SIZE:
                        stats.valid_samples.append(perm_str)
                else:
                    stats.invalid_count += 1
                    stats.by_length_invalid[length] += 1
                    stats.first_char_invalid[perm_str[0]] += 1
                    for pos, ch in enumerate(perm_str):
                        stats.char_pos_invalid[pos][ch] += 1
                    for j in range(len(perm_str) - 1):
                        stats.adjacency_invalid[(perm_str[j], perm_str[j + 1])] += 1
                    if error_cat:
                        stats.error_types[error_cat] += 1
                    if len(stats.error_samples) < self.ERROR_SAMPLE_SIZE:
                        stats.error_samples.append((perm_str, error_msg or ''))
                    if len(stats.invalid_samples) < self.SAMPLE_SIZE:
                        stats.invalid_samples.append(perm_str)

                if count % emit_interval == 0 or count == total:
                    pct = int(count / total * 100) if total else 100
                    self.progress.emit(pct)
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    eta = (total - count) / rate if rate > 0 else 0
                    self.status.emit(
                        f"Processing: {count:,}/{total:,} ({pct}%) | "
                        f"{rate:,.0f} perms/s | ETA: {eta:.1f}s"
                    )
                    self.partial.emit(stats.snapshot())

        self.progress.emit(100)
        self.finished.emit(stats)

    def _check_valid_python(self, code_str: str) -> Tuple[bool, Optional[str], Optional[str]]:
        try:
            compile(code_str, '<perm>', 'exec')
            return True, None, None
        except SyntaxError as e:
            msg = str(e).lower()
            if 'eof' in msg:
                return False, 'EOFError', str(e)
            if 'indent' in msg:
                return False, 'IndentationError', str(e)
            if 'invalid syntax' in msg or 'invalid decimal' in msg:
                return False, 'SyntaxError', str(e)
            if 'unexpected' in msg:
                return False, 'UnexpectedToken', str(e)
            return False, type(e).__name__, str(e)
        except ValueError:
            return False, 'ValueError', 'ValueError'
        except Exception as e:
            return False, type(e).__name__, str(e)


# ═══════════════════════════════════════════════════════════════
#  THEME
# ═══════════════════════════════════════════════════════════════

class Theme:
    DARK = {
        'bg': '#1e1e2e', 'card_bg': '#2a2a3e', 'text': '#e0e0e0',
        'accent': '#89b4fa', 'green': '#a6e3a1', 'red': '#f38ba8',
        'orange': '#fab387', 'blue': '#89b4fa', 'fig_bg': '#1e1e2e',
        'axis_bg': '#2a2a3e', 'grid': '#3a3a4e',
    }
    LIGHT = {
        'bg': '#fafafa', 'card_bg': '#f5f5f5', 'text': '#333333',
        'accent': '#1976d2', 'green': '#4caf50', 'red': '#f44336',
        'orange': '#ff9800', 'blue': '#2196f3', 'fig_bg': '#fafafa',
        'axis_bg': '#ffffff', 'grid': '#e0e0e0',
    }

    @staticmethod
    def apply(app, dark=True):
        t = Theme.DARK if dark else Theme.LIGHT
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(t['bg']))
        palette.setColor(QPalette.WindowText, QColor(t['text']))
        palette.setColor(QPalette.Base, QColor(t['card_bg']))
        palette.setColor(QPalette.AlternateBase, QColor(t['bg']))
        palette.setColor(QPalette.Text, QColor(t['text']))
        palette.setColor(QPalette.Button, QColor(t['card_bg']))
        palette.setColor(QPalette.ButtonText, QColor(t['text']))
        palette.setColor(QPalette.Highlight, QColor(t['accent']))
        palette.setColor(QPalette.HighlightedText, QColor('#ffffff'))
        app.setPalette(palette)
        return t


# ═══════════════════════════════════════════════════════════════
#  ANIMATED CANVAS
# ═══════════════════════════════════════════════════════════════

class AnimatedCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=(7, 5), theme=None):
        self.theme = theme or Theme.LIGHT
        self.fig = Figure(figsize=figsize, dpi=100, facecolor=self.theme['fig_bg'])
        super().__init__(self.fig)
        self.setParent(parent)
        self._current: Dict[str, float] = {}
        self._target: Dict[str, float] = {}
        self._render_fn = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def set_render(self, fn):
        self._render_fn = fn

    def set_targets(self, targets: Dict[str, float]):
        for k in targets:
            if k not in self._current:
                self._current[k] = 0.0
        self._target = targets
        if not self._timer.isActive():
            self._timer.start()

    def show_message(self, text, color=None):
        self._timer.stop()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(self.theme['axis_bg'])
        ax.text(0.5, 0.5, text, ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color=color or self.theme['text'], wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        self.fig.tight_layout(); self.draw()

    def _tick(self):
        done = True
        for key in self._target:
            cur = self._current.get(key, 0)
            tgt = self._target[key]
            diff = tgt - cur
            if abs(diff) > 0.5:
                self._current[key] = cur + diff * 0.18
                done = False
            else:
                self._current[key] = tgt
        if self._render_fn:
            self._render_fn(self._current)
            self.draw_idle()
        if done:
            self._timer.stop()

    def apply_theme(self, theme):
        self.theme = theme
        self.fig.set_facecolor(theme['fig_bg'])
        if self._render_fn:
            self._render_fn(self._current)
            self.draw_idle()


# ═══════════════════════════════════════════════════════════════
#  VISUALIZATION WIDGETS
# ═══════════════════════════════════════════════════════════════

class OverviewWidget(QWidget):
    def __init__(self, parent=None, theme=None):
        self.theme = theme or Theme.LIGHT
        super().__init__(parent)
        layout = QHBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = AnimatedCanvas(self, figsize=(5, 5), theme=self.theme)
        self.canvas.set_render(self._render_pie)
        layout.addWidget(self.canvas, 1)

        card = QFrame()
        card.setStyleSheet(f"QFrame{{background:{self.theme['card_bg']};border-radius:8px;}}")
        cl = QVBoxLayout(card); cl.setContentsMargins(20, 20, 20, 20); cl.setSpacing(8)
        title = QLabel("📊 Summary"); title.setStyleSheet(
            f"font-size:18px;font-weight:bold;color:{self.theme['text']};border:none;")
        cl.addWidget(title)

        self.labels: Dict[str, QLabel] = {}
        for key in ['total', 'valid', 'invalid', 'valid_pct', 'invalid_pct', 'top_error']:
            l = QLabel("—"); l.setStyleSheet(
                f"font-size:14px;color:{self.theme['text']};border:none;")
            self.labels[key] = l; cl.addWidget(l)
        cl.addStretch()
        layout.addWidget(card, 1)

    def _render_pie(self, state):
        v = state.get('valid', 0)
        iv = state.get('invalid', 0)
        t = v + iv
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.set_facecolor(self.theme['axis_bg'])
        if t < 0.5:
            ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color=self.theme['text'])
            ax.set_xticks([]); ax.set_yticks([]); return
        vp = v / t * 100 if t else 0
        ip = iv / t * 100 if t else 0
        sizes = [v, iv]
        labels = [f'Valid\n{int(v):,} ({vp:.1f}%)', f'Invalid\n{int(iv):,} ({ip:.1f}%)']
        colors = [self.theme['green'], self.theme['red']]
        explode = (0.05, 0) if v > 0 else (0, 0.05)
        wedges, texts = ax.pie(
            sizes, labels=labels, colors=colors, explode=explode,
            startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold', 'color': self.theme['text']},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        ax.set_title('Valid vs Invalid Python Code', fontsize=16, fontweight='bold',
                      color=self.theme['text'], pad=20)
        centre = plt.Circle((0, 0), 0.55, fc=self.theme['fig_bg'], ec='white', lw=2)
        ax.add_artist(centre)
        ax.text(0, 0, f'{int(t):,}\nTotal', ha='center', va='center',
                fontsize=14, fontweight='bold', color=self.theme['text'])
        self.canvas.fig.tight_layout()

    def update_data(self, r: PermStats):
        v, iv = r.valid_count, r.invalid_count
        t = r.total if r.total else v + iv
        self.canvas.set_targets({'valid': v, 'invalid': iv})
        vp = v / t * 100 if t else 0
        ip = iv / t * 100 if t else 0
        self.labels['total'].setText(f"Total permutations:  {t:,}")
        self.labels['valid'].setText(
            f"<span style='color:{self.theme['green']};font-weight:bold'>✅ Valid:  {v:,}</span>")
        self.labels['invalid'].setText(
            f"<span style='color:{self.theme['red']};font-weight:bold'>❌ Invalid:  {iv:,}</span>")
        self.labels['valid_pct'].setText(f"Valid %:  {vp:.2f}%")
        self.labels['invalid_pct'].setText(f"Invalid %:  {ip:.2f}%")
        if r.error_types:
            te = r.error_types.most_common(1)[0]
            self.labels['top_error'].setText(
                f"<span style='color:{self.theme['orange']}'>Top error: {te[0]} ({te[1]})</span>")
        else:
            self.labels['top_error'].setText("Top error: None")

    def apply_theme(self, theme):
        self.theme = theme
        self.canvas.apply_theme(theme)
        card = self.findChild(QFrame)
        if card:
            card.setStyleSheet(f"QFrame{{background:{theme['card_bg']};border-radius:8px;}}")
        for l in self.labels.values():
            l.setStyleSheet(f"font-size:14px;color:{theme['text']};border:none;")


class LengthWidget(QWidget):
    def __init__(self, parent=None, theme=None):
        self.theme = theme or Theme.LIGHT
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.bar_canvas = AnimatedCanvas(self, figsize=(7, 4), theme=self.theme)
        self.bar_canvas.set_render(self._render_bar)
        layout.addWidget(self.bar_canvas)
        self.ratio_canvas = AnimatedCanvas(self, figsize=(7, 3), theme=self.theme)
        self.ratio_canvas.set_render(self._render_ratio)
        layout.addWidget(self.ratio_canvas)
        self._max_len = 1

    def _render_bar(self, state):
        self.bar_canvas.fig.clear()
        ax = self.bar_canvas.fig.add_subplot(111)
        ax.set_facecolor(self.theme['axis_bg'])
        n = self._max_len
        x = np.arange(1, n + 1)
        vv = [state.get(f'v{i}', 0) for i in range(n)]
        iv = [state.get(f'i{i}', 0) for i in range(n)]
        ax.bar(x, vv, label='Valid', color=self.theme['green'], edgecolor='white', linewidth=0.5)
        ax.bar(x, iv, bottom=vv, label='Invalid', color=self.theme['red'], edgecolor='white', linewidth=0.5)
        for i, (v, inv) in enumerate(zip(vv, iv)):
            tot = v + inv
            if tot > 0.5:
                ax.text(x[i], tot + 0.3, f'{int(tot)}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color=self.theme['text'])
            if v > 1:
                ax.text(x[i], v / 2, f'{int(v)}', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
            if inv > 1:
                ax.text(x[i], v + inv / 2, f'{int(inv)}', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
        ax.set_xlabel('Permutation Length', fontsize=12, color=self.theme['text'])
        ax.set_ylabel('Count', fontsize=12, color=self.theme['text'])
        ax.set_title('Stacked: Valid & Invalid by Length', fontsize=14, fontweight='bold',
                      color=self.theme['text'])
        ax.set_xticks(x); ax.tick_params(colors=self.theme['text'])
        ax.legend(fontsize=11, loc='upper right',
                  facecolor=self.theme['card_bg'], edgecolor=self.theme['grid'],
                  labelcolor=self.theme['text'])
        max_h = max((v + i for v, i in zip(vv, iv)), default=1)
        ax.set_ylim(0, max(max_h * 1.15, 1))
        ax.grid(axis='y', alpha=0.2, color=self.theme['grid'])
        self.bar_canvas.fig.tight_layout()

    def _render_ratio(self, state):
        self.ratio_canvas.fig.clear()
        ax = self.ratio_canvas.fig.add_subplot(111)
        ax.set_facecolor(self.theme['axis_bg'])
        n = self._max_len
        x = np.arange(1, n + 1)
        ratios = []
        for i in range(n):
            v = state.get(f'v{i}', 0); inv = state.get(f'i{i}', 0)
            t = v + inv
            ratios.append((v / t * 100) if t > 0 else 0)
        ax.plot(x, ratios, 'o-', color=self.theme['blue'], linewidth=2.5, markersize=10,
                markerfacecolor=self.theme['blue'], markeredgecolor='white', markeredgewidth=2)
        ax.fill_between(x, ratios, alpha=0.15, color=self.theme['blue'])
        ax.axhline(y=50, color=self.theme['text'], linestyle='--', alpha=0.4)
        for i, rt in enumerate(ratios):
            ax.annotate(f'{rt:.1f}%', (x[i], rt), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=11, fontweight='bold',
                        color=self.theme['blue'])
        ax.set_xlabel('Permutation Length', fontsize=12, color=self.theme['text'])
        ax.set_ylabel('Valid %', fontsize=12, color=self.theme['text'])
        ax.set_title('Valid Percentage Trend by Length', fontsize=14, fontweight='bold',
                      color=self.theme['text'])
        ax.set_xticks(x); ax.tick_params(colors=self.theme['text'])
        ax.set_ylim(-5, 110)
        ax.grid(axis='y', alpha=0.2, color=self.theme['grid'])
        self.ratio_canvas.fig.tight_layout()

    def update_data(self, r: PermStats):
        n = r.max_length
        self._max_len = n
        targets = {}
        for i in range(1, n + 1):
            targets[f'v{i-1}'] = r.by_length_valid.get(i, 0)
            targets[f'i{i-1}'] = r.by_length_invalid.get(i, 0)
        self.bar_canvas.set_targets(targets)
        self.ratio_canvas.set_targets(targets)

    def apply_theme(self, theme):
        self.theme = theme
        self.bar_canvas.apply_theme(theme)
        self.ratio_canvas.apply_theme(theme)


class ErrorWidget(QWidget):
    def __init__(self, parent=None, theme=None):
        self.theme = theme or Theme.LIGHT
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.bar_canvas = AnimatedCanvas(self, figsize=(7, 4), theme=self.theme)
        self.bar_canvas.set_render(self._render_bar)
        layout.addWidget(self.bar_canvas)
        self.sample_canvas = AnimatedCanvas(self, figsize=(7, 3), theme=self.theme)
        layout.addWidget(self.sample_canvas)
        self._labels: List[str] = []

    def _render_bar(self, state):
        self.bar_canvas.fig.clear()
        ax = self.bar_canvas.fig.add_subplot(111)
        ax.set_facecolor(self.theme['axis_bg'])
        if not self._labels:
            ax.text(0.5, 0.5, 'No errors yet...', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color=self.theme['text'])
            ax.set_xticks([]); ax.set_yticks([])
            self.bar_canvas.fig.tight_layout(); return
        values = [state.get(l, 0) for l in self._labels]
        colors = plt.cm.Set3(np.linspace(0.15, 0.85, len(self._labels)))
        bars = ax.barh(self._labels, values, color=colors, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0.5:
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2.,
                        f'{int(val)}', va='center', fontsize=12, fontweight='bold',
                        color=self.theme['text'])
        ax.set_xlabel('Count', fontsize=12, color=self.theme['text'])
        ax.set_title('Error Type Distribution', fontsize=14, fontweight='bold',
                      color=self.theme['text'])
        ax.invert_yaxis(); ax.tick_params(colors=self.theme['text'])
        max_v = max(values) if values else 1
        ax.set_xlim(0, max(max_v * 1.2, 1))
        ax.grid(axis='x', alpha=0.2, color=self.theme['grid'])
        self.bar_canvas.fig.tight_layout()

    def update_data(self, r: PermStats):
        self._labels = [it[0] for it in r.error_types.most_common()]
        targets = {l: r.error_types[l] for l in self._labels}
        self.bar_canvas.set_targets(targets)
        self.sample_canvas._timer.stop()
        self.sample_canvas.fig.clear()
        ax2 = self.sample_canvas.fig.add_subplot(111)
        ax2.set_facecolor(self.theme['axis_bg']); ax2.axis('off')
        ax2.set_title('Sample Error Messages (first 200)', fontsize=12, fontweight='bold',
                       color=self.theme['text'])
        samples = r.error_samples[:200]
        if not samples:
            ax2.text(0.5, 0.5, 'No errors to display', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=13, color=self.theme['green'])
        else:
            lines = []
            for perm, msg in samples:
                short = msg[:100].replace('\n', ' ')
                lines.append(f"'{perm}' → {short}")
            ax2.text(0.02, 0.95, '\n'.join(lines), va='top', ha='left',
                     transform=ax2.transAxes, fontsize=7, fontfamily='monospace', wrap=True,
                     bbox=dict(boxstyle='round', facecolor=self.theme['card_bg'], alpha=0.8,
                               edgecolor=self.theme['grid']))
        self.sample_canvas.fig.tight_layout()
        self.sample_canvas.draw()

    def apply_theme(self, theme):
        self.theme = theme
        self.bar_canvas.apply_theme(theme)
        self.sample_canvas.apply_theme(theme)


class HeatmapWidget(QWidget):
    def __init__(self, parent=None, theme=None):
        self.theme = theme or Theme.LIGHT
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.valid_canvas = AnimatedCanvas(self, figsize=(7, 3.5), theme=self.theme)
        self.valid_canvas.set_render(lambda s: self._render_heat(self.valid_canvas, s, 'Greens', 'Valid: Char × Position'))
        layout.addWidget(self.valid_canvas)
        self.invalid_canvas = AnimatedCanvas(self, figsize=(7, 3.5), theme=self.theme)
        self.invalid_canvas.set_render(lambda s: self._render_heat(self.invalid_canvas, s, 'Reds', 'Invalid: Char × Position'))
        layout.addWidget(self.invalid_canvas)
        self._chars: List[str] = []
        self._max_len = 1

    def _render_heat(self, canvas, state, cmap, title):
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        ax.set_facecolor(self.theme['axis_bg'])
        chars = self._chars; n = self._max_len
        matrix = np.zeros((len(chars), n))
        for pos in range(n):
            for ci, ch in enumerate(chars):
                matrix[ci, pos] = state.get(f'{ci}_{pos}', 0)
        if matrix.max() < 0.5:
            ax.text(0.5, 0.5, 'No data yet...', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color=self.theme['text'])
            ax.set_xticks([]); ax.set_yticks([])
            canvas.fig.tight_layout(); return
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='bilinear')
        ax.set_xticks(range(n)); ax.set_xticklabels([f'Pos {i}' for i in range(n)], fontsize=10)
        ax.set_yticks(range(len(chars))); ax.set_yticklabels(chars, fontsize=12, fontweight='bold')
        ax.tick_params(colors=self.theme['text'])
        ax.set_title(title, fontsize=13, fontweight='bold', color=self.theme['text'])
        mx = matrix.max()
        for i in range(len(chars)):
            for j in range(n):
                val = matrix[i, j]
                if val > 0.5:
                    color = 'white' if val > mx * 0.5 else 'black'
                    ax.text(j, i, f'{int(val)}', ha='center', va='center',
                            fontsize=9, color=color, fontweight='bold')
        canvas.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        canvas.fig.tight_layout()

    def update_data(self, r: PermStats):
        chars = list(r.characters); n = r.max_length
        self._chars = chars; self._max_len = n
        for canvas, data in [(self.valid_canvas, r.char_pos_valid),
                             (self.invalid_canvas, r.char_pos_invalid)]:
            targets = {}
            for pos in range(n):
                for ci, ch in enumerate(chars):
                    targets[f'{ci}_{pos}'] = data.get(pos, {}).get(ch, 0)
            canvas.set_targets(targets)

    def apply_theme(self, theme):
        self.theme = theme
        self.valid_canvas.apply_theme(theme)
        self.invalid_canvas.apply_theme(theme)


class NetworkWidget(QWidget):
    def __init__(self, parent=None, theme=None):
        self.theme = theme or Theme.LIGHT
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = AnimatedCanvas(self, figsize=(8, 6), theme=self.theme)
        self.canvas.set_render(self._render)
        layout.addWidget(self.canvas)
        self._chars: List[str] = []

    def _render(self, state):
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.set_facecolor(self.theme['fig_bg'])
        chars = self._chars
        if not chars:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color=self.theme['text'])
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas.fig.tight_layout(); return
        n = len(chars)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
        pos = {ch: (np.cos(angles[i]), np.sin(angles[i])) for i, ch in enumerate(chars)}
        edges = {}
        for key, w in state.items():
            if isinstance(key, tuple) and isinstance(w, (int, float)) and w > 0.1:
                edges[key] = w
        max_w = max(edges.values()) if edges else 1
        for (c1, c2), w in edges.items():
            x1, y1 = pos[c1]; x2, y2 = pos[c2]
            alpha = min(w / max_w, 1.0) if max_w > 0 else 0
            lw = 1 + 5 * alpha
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=self.theme['blue'],
                                        alpha=0.3 + 0.7 * alpha, lw=lw,
                                        connectionstyle='arc3,rad=0.15'))
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if w > 0.5:
                ax.text(mx, my, f'{int(w)}', fontsize=8, ha='center', va='center',
                        color=self.theme['blue'], fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  alpha=0.7, edgecolor='none'))
        for ch in chars:
            x, y = pos[ch]
            circle = plt.Circle((x, y), 0.08, color=self.theme['green'],
                                ec=self.theme['text'], lw=2, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, ch, ha='center', va='center', fontsize=14,
                    fontweight='bold', color='white', zorder=6)
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal')
        ax.set_title('Character Adjacency Network (Valid)', fontsize=14, fontweight='bold',
                      color=self.theme['text'])
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.fig.tight_layout()

    def update_data(self, r: PermStats):
        self._chars = list(r.characters)
        targets = {}
        for pair, w in r.adjacency_valid.items():
            targets[pair] = w
        for k in list(self.canvas._current.keys()):
            if k not in targets:
                targets[k] = 0
        self.canvas.set_targets(targets)

    def apply_theme(self, theme):
        self.theme = theme
        self.canvas.apply_theme(theme)


class TreemapWidget(QWidget):
    def __init__(self, parent=None, theme=None):
        self.theme = theme or Theme.LIGHT
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = AnimatedCanvas(self, figsize=(8, 5), theme=self.theme)
        self.canvas.set_render(self._render)
        layout.addWidget(self.canvas)
        self._chars: List[str] = []

    def _render(self, state):
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.set_facecolor(self.theme['axis_bg'])
        chars = self._chars
        fv = {ch: state.get(f'v_{ch}', 0) for ch in chars}
        fi = {ch: state.get(f'i_{ch}', 0) for ch in chars}
        totals = {ch: fv[ch] + fi[ch] for ch in chars}
        grand = sum(totals.values())
        if grand < 0.5:
            ax.text(0.5, 0.5, 'No data yet...', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color=self.theme['text'])
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas.fig.tight_layout(); return
        rects = self._squarify(totals, 0, 0, 1, 1)
        palette = plt.cm.Set2(np.linspace(0, 1, len(chars)))
        for (ch, x, y, w, h), color in zip(rects, palette):
            v = fv[ch]; iv = fi[ch]; t = v + iv
            if t < 0.5: continue
            vf = v / t if t > 0 else 0
            ax.add_patch(plt.Rectangle((x, y), w, h * vf, facecolor=mcolors.to_hex(color),
                                       alpha=0.85, edgecolor='white', linewidth=2))
            ax.add_patch(plt.Rectangle((x, y + h * vf), w, h * (1 - vf),
                                       facecolor=mcolors.to_hex(color), alpha=0.35,
                                       edgecolor='white', linewidth=2))
            cx, cy = x + w / 2, y + h / 2
            fs = max(8, min(20, int(w * 30)))
            ax.text(cx, cy, f"'{ch}'\n{int(t)}", ha='center', va='center',
                    fontsize=fs, fontweight='bold', color='#333')
            if v > 0.5:
                ax.text(cx, y + h * vf * 0.5, f'✅{int(v)}', ha='center', va='center',
                        fontsize=max(6, fs - 3), color='#1B5E20')
            if iv > 0.5:
                ax.text(cx, y + h * vf + h * (1 - vf) * 0.5, f'❌{int(iv)}',
                        ha='center', va='center', fontsize=max(6, fs - 3), color='#B71C1C')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title('Treemap: First Character Distribution\n(solid=valid, faded=invalid)',
                     fontsize=14, fontweight='bold', color=self.theme['text'])
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.fig.tight_layout()

    def _squarify(self, data, x, y, w, h):
        items = sorted(data.items(), key=lambda kv: -kv[1])
        total = sum(v for _, v in items)
        if total == 0: return []
        rects = []
        cx, cy, cw, ch = x, y, w, h
        remaining = list(items)
        while remaining:
            if len(remaining) == 1:
                ch_item, val = remaining[0]
                rects.append((ch_item, cx, cy, cw, ch)); break
            row = []; row_sum = 0
            for item in remaining:
                if row and (row_sum + item[1]) > total * (min(cw, ch) / max(cw, ch)):
                    break
                row.append(item); row_sum += item[1]
            remaining = [it for it in remaining if it not in row]
            if cw >= ch:
                row_w = (row_sum / total) * cw if total else 0; ry = cy
                for ch_item, val in row:
                    rh = (val / row_sum) * ch if row_sum else 0
                    rects.append((ch_item, cx, ry, row_w, rh)); ry += rh
                cx += row_w; cw -= row_w; total -= row_sum
            else:
                row_h = (row_sum / total) * ch if total else 0; rx = cx
                for ch_item, val in row:
                    rw = (val / row_sum) * cw if row_sum else 0
                    rects.append((ch_item, rx, cy, rw, row_h)); rx += rw
                cy += row_h; ch -= row_h; total -= row_sum
        return rects

    def update_data(self, r: PermStats):
        self._chars = list(r.characters)
        targets = {}
        for ch in self._chars:
            targets[f'v_{ch}'] = r.first_char_valid.get(ch, 0)
            targets[f'i_{ch}'] = r.first_char_invalid.get(ch, 0)
        self.canvas.set_targets(targets)

    def apply_theme(self, theme):
        self.theme = theme
        self.canvas.apply_theme(theme)


# ═══════════════════════════════════════════════════════════════
#  HIGH-PERFORMANCE SAMPLE TABLE (QTableView + QAbstractTableModel)
# ═══════════════════════════════════════════════════════════════

class SampleTableWidget(QWidget):
    """Uses QTableView + QAbstractTableModel for 10,000+ row performance."""

    def __init__(self, title: str, parent=None, theme=None):
        self.theme = theme or Theme.LIGHT
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(4)

        header = QHBoxLayout()
        self.label = QLabel(title)
        self.label.setStyleSheet(f"font-weight:bold;padding:4px;color:{self.theme['text']};")
        header.addWidget(self.label)
        header.addStretch()
        self.count_label = QLabel("")
        self.count_label.setStyleSheet(f"color:{self.theme['text']};font-size:12px;")
        header.addWidget(self.count_label)
        header.addSpacing(8)
        search_label = QLabel("🔍")
        search_label.setStyleSheet(f"color:{self.theme['text']};")
        header.addWidget(search_label)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter permutations...")
        self.search.setFixedWidth(200)
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        header.addWidget(self.search)
        layout.addLayout(header)

        self.model = SampleModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(2, 50)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setEditTriggers(QTableView.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setSortingEnabled(False)
        self.table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table)

    def update_samples(self, samples: list, total_count: int):
        self.model.set_data(samples)
        n = len(samples)
        if n == total_count:
            self.count_label.setText(f"{total_count:,} samples")
        else:
            self.count_label.setText(f"showing {n:,} of {total_count:,}")
        self.label.setText(self.label.text().split('(')[0].strip())

    def _filter(self, text: str):
        self.model.set_filter(text)
        total = self.model.total_count()
        filtered = self.model.filtered_count()
        if text:
            self.count_label.setText(f"{filtered:,} / {total:,} (filtered)")
        else:
            self.count_label.setText(f"{total:,} samples")

    def _on_double_click(self, index):
        row = index.row()
        perm = self.model.get_perm(row)
        if perm:
            try:
                compile(perm, '<perm>', 'exec')
                QMessageBox.information(self, "Inspection Result",
                    f"✅ '{perm}' is valid Python code!")
            except SyntaxError as e:
                QMessageBox.information(self, "Inspection Result",
                    f"❌ '{perm}' is invalid.\n\nError: {e}")

    def apply_theme(self, theme):
        self.theme = theme
        self.label.setStyleSheet(f"font-weight:bold;padding:4px;color:{theme['text']};")
        self.count_label.setStyleSheet(f"color:{theme['text']};font-size:12px;")
        self.table.setStyleSheet(
            f"QTableView{{alternate-background-color:{theme['bg']};"
            f"background:{theme['card_bg']};color:{theme['text']};"
            f"gridline-color:{theme['grid']};border:none;}}"
            f"QHeaderView::section{{background:{theme['card_bg']};color:{theme['text']};"
            f"border:1px solid {theme['grid']};padding:4px;font-weight:bold;}}"
            f"QLineEdit{{background:{theme['card_bg']};color:{theme['text']};"
            f"border:1px solid {theme['grid']};border-radius:3px;padding:4px;}}")


# ═══════════════════════════════════════════════════════════════
#  PRESETS
# ═══════════════════════════════════════════════════════════════

PRESETS = {
    "Letters (a-d)": "abcd",
    "Letters (a-f)": "abcdef",
    "Letters (a-h)": "abcdefgh",
    "Digits (1-4)": "1234",
    "Digits (0-4)": "01234",
    "Operators": "+-*=",
    "Punctuation": "()[]{}",
    "Mixed": "ab12",
    "Python-ish": "def()",
    "Custom": "",
}


# ═══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Permutation → Python Code Validator (10K Samples)")
        self.setGeometry(60, 60, 1450, 920)
        self.worker: Optional[PermutationWorker] = None
        self.results: Optional[PermStats] = None
        self._dark = True
        self.theme = Theme.apply(QApplication.instance(), dark=self._dark)
        self._init_ui()
        self._apply_theme()

    def _init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central); main_layout.setSpacing(6)

        # ── Header ──
        header = QFrame()
        header.setObjectName("header")
        hl = QHBoxLayout(header)
        title = QLabel("🔬 Permutation → Python Code Validator — 10K Sample Capacity")
        title.setObjectName("title")
        title.setStyleSheet("font-size:20px;font-weight:bold;padding:10px;")
        hl.addWidget(title); hl.addStretch()

        self.theme_btn = QPushButton("🌙 Dark")
        self.theme_btn.setFixedWidth(80)
        self.theme_btn.clicked.connect(self._toggle_theme)
        hl.addWidget(self.theme_btn)
        main_layout.addWidget(header)

        # ── Config ──
        cfg = QGroupBox("⚙️ Configuration")
        cl = QHBoxLayout(cfg)
        cl.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for name, chars in PRESETS.items():
            self.preset_combo.addItem(name, chars)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        cl.addWidget(self.preset_combo)
        cl.addWidget(QLabel("n (max):"))
        self.n_spin = QSpinBox(); self.n_spin.setMinimum(1); self.n_spin.setMaximum(10); self.n_spin.setValue(4)
        self.n_spin.valueChanged.connect(self._preview)
        cl.addWidget(self.n_spin)
        cl.addWidget(QLabel("Chars:"))
        self.chars_edit = QLineEdit("abcd")
        self.chars_edit.textChanged.connect(self._preview)
        cl.addWidget(self.chars_edit, 1)
        self.gen_btn = QPushButton("⚡ Generate")
        self.gen_btn.setObjectName("gen")
        self.gen_btn.clicked.connect(self._generate)
        cl.addWidget(self.gen_btn)
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        cl.addWidget(self.pause_btn)
        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        cl.addWidget(self.cancel_btn)
        self.export_btn = QPushButton("💾 Export")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)
        cl.addWidget(self.export_btn)
        main_layout.addWidget(cfg)

        # ── Progress Panel ──
        prog_frame = QFrame()
        pl = QHBoxLayout(prog_frame); pl.setContentsMargins(4, 4, 4, 4)
        self.info_label = QLabel("Ready")
        self.info_label.setStyleSheet("font-style:italic;")
        pl.addWidget(self.info_label)
        pl.addStretch()
        self.rate_label = QLabel("")
        pl.addWidget(self.rate_label)
        main_layout.addWidget(prog_frame)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # ── Splitter ──
        splitter = QSplitter(Qt.Horizontal)
        left = QFrame(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        self.viz_tabs = QTabWidget()
        self.overview = OverviewWidget(theme=self.theme)
        self.viz_tabs.addTab(self.overview, "📊 Overview")
        self.length_w = LengthWidget(theme=self.theme)
        self.viz_tabs.addTab(self.length_w, "📏 By Length")
        self.error_w = ErrorWidget(theme=self.theme)
        self.viz_tabs.addTab(self.error_w, "⚠️ Errors")
        self.heatmap_w = HeatmapWidget(theme=self.theme)
        self.viz_tabs.addTab(self.heatmap_w, "🗺️ Heatmap")
        self.network_w = NetworkWidget(theme=self.theme)
        self.viz_tabs.addTab(self.network_w, "🕸️ Network")
        self.treemap_w = TreemapWidget(theme=self.theme)
        self.viz_tabs.addTab(self.treemap_w, "📐 Treemap")
        ll.addWidget(self.viz_tabs)
        splitter.addWidget(left)
        right = QFrame(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        self.data_tabs = QTabWidget()
        self.valid_tbl = SampleTableWidget("✅ Valid Samples", theme=self.theme)
        self.data_tabs.addTab(self.valid_tbl, "✅ Valid")
        self.invalid_tbl = SampleTableWidget("❌ Invalid Samples", theme=self.theme)
        self.data_tabs.addTab(self.invalid_tbl, "❌ Invalid")
        rl.addWidget(self.data_tabs)
        splitter.addWidget(right)
        splitter.setSizes([880, 570])
        main_layout.addWidget(splitter, 1)
        self.statusBar().showMessage("Ready — select a preset or enter characters, then click Generate")

        # ── Shortcuts ──
        QShortcut("Return", self).activated.connect(self._generate)
        QShortcut("Esc", self).activated.connect(self._cancel)
        QShortcut("Ctrl+S", self).activated.connect(self._export)
        QShortcut("Ctrl+D", self).activated.connect(self._toggle_theme)
        QShortcut("Space", self).activated.connect(self._toggle_pause)
        self._preview()

    def _apply_theme(self):
        t = self.theme
        header = self.findChild(QFrame, "header")
        header.setStyleSheet(
            f"QFrame#header{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {'#1a237e' if self._dark else '#1565c2'},"
            f"stop:1 {'#0d47a1' if self._dark else '#1976d2'});"
            f"border-radius:8px;}}"
            f"QLabel#title{{color:white;border:none;}}")
        for gb in self.findChildren(QGroupBox):
            gb.setStyleSheet(
                f"QGroupBox{{font-weight:bold;color:{t['text']};"
                f"border:1px solid {t['grid']};border-radius:6px;"
                f"margin-top:8px;padding-top:12px;}}")
        self.gen_btn.setStyleSheet(
            f"QPushButton#gen{{font-weight:bold;padding:8px 16px;"
            f"background:{t['green']};color:white;border:none;border-radius:4px;}}"
            f"QPushButton#gen:hover{{opacity:0.9;}}"
            f"QPushButton#gen:disabled{{background:#555;}}")
        self.cancel_btn.setStyleSheet(
            f"QPushButton{{padding:8px 12px;background:{t['red']};color:white;"
            f"border:none;border-radius:4px;}}"
            f"QPushButton:hover{{opacity:0.9;}}"
            f"QPushButton:disabled{{background:#555;}}")
        self.pause_btn.setStyleSheet(
            f"QPushButton{{padding:8px 12px;background:{t['orange']};color:white;"
            f"border:none;border-radius:4px;}}"
            f"QPushButton:hover{{opacity:0.9;}}"
            f"QPushButton:disabled{{background:#555;}}")
        self.export_btn.setStyleSheet(
            f"QPushButton{{padding:8px 12px;background:{t['blue']};color:white;"
            f"border:none;border-radius:4px;}}"
            f"QPushButton:hover{{opacity:0.9;}}"
            f"QPushButton:disabled{{background:#555;}}")
        self.theme_btn.setText("☀️ Light" if self._dark else "🌙 Dark")
        self.progress_bar.setStyleSheet(
            f"QProgressBar{{border-radius:4px;text-align:center;height:22px;"
            f"background:{t['card_bg']};color:{t['text']};}}"
            f"QProgressBar::chunk{{border-radius:3px;background:{t['blue']};}}")
        tab_style = (
            f"QTabWidget::pane{{border:1px solid {t['grid']};border-radius:4px;"
            f"background:{t['bg']};}}"
            f"QTabBar::tab{{padding:8px 16px;font-weight:bold;color:{t['text']};"
            f"background:{t['card_bg']};border:1px solid {t['grid']};"
            f"border-bottom:none;border-top-left-radius:4px;border-top-right-radius:4px;}}"
            f"QTabBar::tab:selected{{background:{t['bg']};color:{t['accent']};}}")
        self.viz_tabs.setStyleSheet(tab_style)
        self.data_tabs.setStyleSheet(tab_style)
        for tbl_w in [self.valid_tbl, self.invalid_tbl]:
            tbl_w.apply_theme(t)
        self.info_label.setStyleSheet(f"font-style:italic;color:{t['text']};")
        self.rate_label.setStyleSheet(f"color:{t['text']};")
        for w in [self.overview, self.length_w, self.error_w,
                  self.heatmap_w, self.network_w, self.treemap_w]:
            w.apply_theme(t)

    def _on_preset(self, idx):
        chars = self.preset_combo.itemData(idx)
        if chars:
            self.chars_edit.setText(chars)

    def _preview(self):
        n = self.n_spin.value(); chars = self.chars_edit.text().strip()
        if chars:
            an = min(n, len(chars))
            total = sum(math.perm(len(chars), l) for l in range(1, an + 1))
            self.info_label.setText(
                f"📋 Chars: '{chars}' | Max length: {an} | Total permutations: {total:,} | "
                f"Sample capacity: {min(total, 10000):,}")

    def _generate(self):
        n = self.n_spin.value(); chars = self.chars_edit.text().strip()
        if not chars:
            QMessageBox.warning(self, "Warning", "Please enter at least one character."); return
        an = min(n, len(chars))
        total = sum(math.perm(len(chars), l) for l in range(1, an + 1))
        if total > 500000:
            reply = QMessageBox.question(self, "Large Dataset",
                f"This will stream {total:,} permutations.\nIt may take a while.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No: return
        self.gen_btn.setEnabled(False)
        self.pause_btn.setEnabled(True); self.pause_btn.setText("⏸ Pause")
        self.cancel_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.statusBar().showMessage("Processing...")
        self.worker = PermutationWorker(chars, n)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self._on_status)
        self.worker.partial.connect(self._on_partial)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_status(self, msg):
        self.statusBar().showMessage(msg)
        if 'perms/s' in msg:
            parts = msg.split('|')
            self.info_label.setText(parts[0].strip())
            self.rate_label.setText('|'.join(parts[1:]).strip())
        else:
            self.info_label.setText(msg)
            self.rate_label.setText("")

    def _toggle_pause(self):
        if self.worker and self.worker.isRunning():
            self.worker._pause = not self.worker._pause
            self.pause_btn.setText("▶ Resume" if self.worker._pause else "⏸ Pause")
            self.statusBar().showMessage("Paused" if self.worker._pause else "Processing...")

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
            if self.worker.isRunning():
                self.worker.terminate()
            self.worker = None
        self._reset_buttons()
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Cancelled")

    def _reset_buttons(self):
        self.gen_btn.setEnabled(True)
        self.pause_btn.setEnabled(False); self.pause_btn.setText("⏸ Pause")
        self.cancel_btn.setEnabled(False)
        self.export_btn.setEnabled(self.results is not None)

    def _on_partial(self, r: PermStats):
        self.overview.update_data(r)
        self.length_w.update_data(r)
        self.error_w.update_data(r)
        self.heatmap_w.update_data(r)
        self.network_w.update_data(r)
        self.treemap_w.update_data(r)
        self.valid_tbl.update_samples(r.valid_samples, r.valid_count)
        self.invalid_tbl.update_samples(r.invalid_samples, r.invalid_count)

    def _on_finished(self, r: PermStats):
        self._on_partial(r)
        self.results = r
        self._reset_buttons()
        self.progress_bar.setVisible(False)
        v, iv, t = r.valid_count, r.invalid_count, r.total
        nv = len(r.valid_samples)
        ni = len(r.invalid_samples)
        self.statusBar().showMessage(
            f"✅ Done: {v:,} valid, {iv:,} invalid, {t:,} total | "
            f"Samples: {nv:,} valid, {ni:,} invalid | Memory: stats only")
        self.rate_label.setText("")

    def _toggle_theme(self):
        self._dark = not self._dark
        self.theme = Theme.apply(QApplication.instance(), dark=self._dark)
        self._apply_theme()

    def _export(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to export. Run first."); return
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder: return
        r = self.results; chars = r.characters

        # Save all figures
        for name, widget in [("overview", self.overview), ("length", self.length_w),
                             ("errors", self.error_w), ("heatmap", self.heatmap_w),
                             ("network", self.network_w), ("treemap", self.treemap_w)]:
            for canvas in widget.findChildren(AnimatedCanvas):
                safe_name = f"{name}_{chars}"
                canvas.fig.savefig(f"{folder}/{safe_name}.png", dpi=150,
                                  facecolor=self.theme['fig_bg'])

        # Save CSV with all samples
        csv_path = f"{folder}/stats_{chars}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Statistic", "Value"])
            writer.writerow(["Characters", chars])
            writer.writerow(["Max length", r.max_length])
            writer.writerow(["Total permutations", r.total])
            writer.writerow(["Valid count", r.valid_count])
            writer.writerow(["Invalid count", r.invalid_count])
            writer.writerow(["Valid %", f"{r.valid_count/r.total*100:.2f}" if r.total else "0"])
            writer.writerow(["Valid samples stored", len(r.valid_samples)])
            writer.writerow(["Invalid samples stored", len(r.invalid_samples)])
            writer.writerow([])
            writer.writerow(["Error Type", "Count"])
            for et, cnt in r.error_types.most_common():
                writer.writerow([et, cnt])
            writer.writerow([])
            writer.writerow(["Length", "Valid", "Invalid", "Valid %"])
            for l in range(1, r.max_length + 1):
                v = r.by_length_valid.get(l, 0)
                iv = r.by_length_invalid.get(l, 0)
                t = v + iv
                writer.writerow([l, v, iv, f"{v/t*100:.2f}" if t else "0"])
            writer.writerow([])
            writer.writerow(["Valid Samples (up to 10,000)"])
            for s in r.valid_samples:
                writer.writerow([s])
            writer.writerow([])
            writer.writerow(["Invalid Samples (up to 10,000)"])
            for s in r.invalid_samples:
                writer.writerow([s])

        QMessageBox.information(self, "Export Complete",
            f"Exported to:\n{folder}\n\nFiles:\n"
            f"• 6 PNG charts (150 DPI)\n"
            f"• stats_{chars}.csv (with {len(r.valid_samples):,} valid + "
            f"{len(r.invalid_samples):,} invalid samples)")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
            if self.worker.isRunning():
                self.worker.terminate()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()