import logging
from datetime import datetime
from pathlib import Path

import colorama
from colorama import Fore, Style

# coloramaを初期化します。autoreset=Trueにすると、printのたびに色がリセットされて便利です。
colorama.init(autoreset=True)

# --- カスタムフォーマッター ---
class KABUFormatter(logging.Formatter):
    """色付け（任意）、レベル名の短縮、ファイル情報の表示を行うフォーマッター"""

    COLOR_MAP = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
    }

    def __init__(self, use_color=True):
        super().__init__()
        self.use_color = use_color

    def format(self, record):
        # タイムスタンプの取得 (コンソールは時刻のみ、ファイルは日付込み)
        time_fmt = "%H:%M:%S" if self.use_color else "%Y-%m-%d %H:%M:%S"
        asctime = self.formatTime(record, time_fmt)

        # レベル名の調整
        levelname = record.levelname
        if levelname == "DEBUG":
            levelname = "DEBG"
        elif levelname == "WARNING":
            levelname = "WARN"

        # ファイル情報の構築 (ファイル名を3文字 + 行番号3桁)
        file_tag = f"{record.filename[:3]}"
        file_info = f"[{file_tag}:{record.lineno:03d}]"

        if self.use_color:
            # 色の取得
            paint_stt = self.COLOR_MAP.get(record.levelno, Fore.WHITE)
            paint_end = Style.RESET_ALL
            colored_levelname = f"{paint_stt}[{levelname}]{paint_end}"
            header = f"[{asctime}] {colored_levelname} {file_info} "
        else:
            header = f"[{asctime}] [{levelname}] {file_info} "

        # メッセージの取得
        message = record.getMessage()
        lines = message.splitlines()
        if not lines:
            return header

        # マルチライン対応: 2行目以降のインデントを揃える
        formatted_msg = header + lines[0]
        if len(lines) > 1:
            # エスケープシーケンスを除いたヘッダーの純粋な長さを計算してパディングを作る
            header_len = len(f"[{asctime}] [{levelname}] {file_info} ")
            padding = " " * header_len
            formatted_msg += "\n" + "\n".join([padding + line for line in lines[1:]])
        return formatted_msg


# --- ロガー取得関数 ---
_kabu_logger = None


def get_logger():
    global _kabu_logger
    if _kabu_logger is None:
        base_logger = logging.getLogger("KABU")
        base_logger.setLevel(logging.INFO)

        if not base_logger.handlers:
            # 1. コンソール出力用 (色あり)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(KABUFormatter(use_color=True))
            base_logger.addHandler(console_handler)

            # 2. ファイル出力用 (色なし・日付付き)
            try:
                log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
                log_dir.mkdir(exist_ok=True)
                log_file = log_dir / f"kabu_{datetime.now().strftime('%Y%m%d')}.log"
                
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setFormatter(KABUFormatter(use_color=False))
                base_logger.addHandler(file_handler)
            except Exception as e:
                # ログ設定自体の失敗で本体を止めないよう最低限の通知
                print(f"⚠️ ログファイルの作成に失敗しました: {e}")

            base_logger.propagate = False
        _kabu_logger = base_logger
    return _kabu_logger
