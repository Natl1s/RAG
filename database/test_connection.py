from sqlalchemy import text

from database.connection import engine


def main():
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT 1")
        )

        print("Database connected:", result.scalar())


if __name__ == "__main__":
    main()