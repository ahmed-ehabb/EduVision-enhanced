"""
Initialize the database with tables and test users
"""
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import Base, User, Teacher, Student, UserRole
from database import engine
import crud
from sqlalchemy.orm import Session
import auth

def init_database():
    """Create all tables and add test users"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created successfully")

    # Create a database session
    db = Session(bind=engine)

    try:
        # Check if users already exist
        existing_teacher = crud.get_user_by_email(db, "teacher@example.com")
        if existing_teacher:
            print("✓ Test users already exist")
            return

        print("\nCreating test users...")

        # Create teacher user
        teacher_user = User(
            email="teacher@example.com",
            password_hash=auth.hash_password("password123"),
            role=UserRole.TEACHER,
            is_active=True
        )
        db.add(teacher_user)
        db.flush()  # Get the user_id

        # Create teacher profile
        teacher = Teacher(
            user_id=teacher_user.user_id,
            full_name="Dr. John Smith",
            department="Computer Science"
        )
        db.add(teacher)

        # Create student user
        student_user = User(
            email="student@example.com",
            password_hash=auth.hash_password("password123"),
            role=UserRole.STUDENT,
            is_active=True
        )
        db.add(student_user)
        db.flush()

        # Create student profile
        student = Student(
            user_id=student_user.user_id,
            full_name="Jane Doe",
            student_number="STU12345",
            major="Computer Science",
            year_of_study=3
        )
        db.add(student)

        # Commit all changes
        db.commit()

        print("✓ Teacher account created:")
        print(f"  Email: teacher@example.com")
        print(f"  Password: password123")
        print(f"  Name: {teacher.full_name}")

        print("\n✓ Student account created:")
        print(f"  Email: student@example.com")
        print(f"  Password: password123")
        print(f"  Name: {student.full_name}")

        print("\n✅ Database initialized successfully!")
        print("\nYou can now login at http://localhost:3000")

    except Exception as e:
        print(f"\n❌ Error creating test users: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    try:
        init_database()
    except Exception as e:
        print(f"\n❌ Database initialization failed: {e}")
        sys.exit(1)
