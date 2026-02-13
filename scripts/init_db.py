from db.session import engine, Base
from db.models import Job

def init_database():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")

if __name__ == "__main__":
    init_database()
