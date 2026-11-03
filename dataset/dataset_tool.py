import argparse
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


REPO_ID = "irlspbru/RusLawOD"
DATA_DIR = Path(__file__).resolve().parent / "data" / "RusLawOD"


def download():
    """Скачивает датасет RusLawOD в локальную папку."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Скачивание RusLawOD...")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=["*.parquet"],
    )

    print(f"Датасет сохранён в {DATA_DIR}")


def check():
    """Проверяет наличие и целостность Parquet-файлов."""

    files = sorted(DATA_DIR.glob("*.parquet"))

    if not files:
        print("Датасет не найден. Запусти download.")
        return

    total_rows = 0

    for file in files:
        try:
            parquet = pq.ParquetFile(file)
            rows = parquet.metadata.num_rows

            total_rows += rows
            print(f"{file.name}: {rows} документов")

        except Exception as e:
            print(f"Ошибка чтения {file.name}: {e}")

    print(f"\nНайдено файлов: {len(files)}")
    print(f"Всего документов: {total_rows}")

def example():
    """Выводит первые 1000 символов первого документа."""

    files = sorted(DATA_DIR.glob("*.parquet"))

    if not files:
        print("Датасет не найден. Запусти download.")
        return

    parquet = pq.ParquetFile(files[0])

    batch = next(parquet.iter_batches(batch_size=1))
    first_document = batch.to_pylist()[0]

    print("Название:", first_document["headingIPS"])
    print("Текст:", first_document["textIPS"][:1000])

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Управление датасетом RusLawOD"
    )

    parser.add_argument(
        "command",
        choices=["download", "check", "example"],
        help="Команда для выполнения"
    )

    args = parser.parse_args()

    if args.command == "download":
        download()

    elif args.command == "check":
        check()

    elif args.command == "example":
        example()