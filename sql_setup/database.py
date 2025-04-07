from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database setup
Base = declarative_base()

class Detection(Base):
    __tablename__ = 'license_plate'
    id = Column(Integer, primary_key=True)
    plate_text = Column(String, nullable=False)  # License plate text
    score = Column(Float, nullable=False)  # Confidence score
    source = Column(String, nullable=True)  # Source of the detection (video path)
    timestamp = Column(DateTime, default=datetime.utcnow)  # Timestamp of detection

def setup_database():
    """Create and connect to the SQLite database."""
    engine = create_engine('sqlite:///license_plates.db', echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def insert_detection_data(session, plate_text, score, source=None):
    """
    Insert detection data into the database.
    Args:
        session: SQLAlchemy session object.
        plate_text (str): Detected license plate text.
        score (float): Confidence score of the detection.
        source (str, optional): Source of the detection (e.g., video path).
    """
    # Validate inputs to prevent database errors
    if plate_text is None or not isinstance(plate_text, str) or not plate_text.strip():
        print("Warning: Attempted to insert None or empty plate_text. Skipping database insert.")
        return

    try:
        # Convert score to float if it's not already
        score = float(score)
    except (ValueError, TypeError):
        print(f"Warning: Invalid score value ({score}). Using default value.")
        score = 0.0

    # Create and add the detection object
    detection = Detection(plate_text=plate_text, score=score, source=source)
    try:
        session.add(detection)
        session.commit()
        print(f"Successfully inserted detection: {plate_text}, score: {score}")
    except Exception as e:
        session.rollback()
        print(f"Error inserting detection: {e}")