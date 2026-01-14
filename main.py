from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, field_validator, EmailStr
from datetime import datetime, timedelta, timezone
import sqlite3
from typing import List, Optional
from enum import Enum
import json
import os
import secrets
from contextlib import contextmanager
import logging
from passlib.context import CryptContext
from jose import JWTError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
DATABASE_URL = os.getenv("DATABASE_URL", "expenses.db")
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(32)
    logger.warning("SECRET_KEY not set in environment. Using auto-generated key. Set SECRET_KEY in production!")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Smart Expense Tracker API",
    description="Track expenses and get AI-powered budget recommendations",
    version="1.0.0"
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database context manager
@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()

# Database initialization
def init_db():
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    email TEXT UNIQUE,
                    name TEXT,
                    monthly_budget REAL DEFAULT 50000,
                    password_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    category TEXT,
                    amount REAL,
                    description TEXT,
                    date DATE DEFAULT CURRENT_DATE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS budgets (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    category TEXT,
                    limit_amount REAL,
                    month INTEGER,
                    year INTEGER,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            ''')
            
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

init_db()

# Enums
class ExpenseCategory(str, Enum):
    FOOD = "Food"
    TRANSPORT = "Transport"
    UTILITIES = "Utilities"
    ENTERTAINMENT = "Entertainment"
    SHOPPING = "Shopping"
    HEALTH = "Health"
    EDUCATION = "Education"
    OTHER = "Other"

# Pydantic models
class UserRegister(BaseModel):
    email: EmailStr
    name: str
    monthly_budget: float = 50000
    password: str
    
    @field_validator('monthly_budget')
    @classmethod
    def budget_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Monthly budget must be positive')
        return v

class User(BaseModel):
    email: EmailStr
    name: str
    monthly_budget: float = 50000
    
    @field_validator('monthly_budget')
    @classmethod
    def budget_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Monthly budget must be positive')
        return v

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    monthly_budget: float

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None

class Expense(BaseModel):
    category: ExpenseCategory
    amount: float
    description: str = ""
    date: Optional[str] = None
    
    @field_validator('amount')
    @classmethod
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v
    
    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v):
        if v is not None:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('Date must be in YYYY-MM-DD format')
        return v

class BudgetSet(BaseModel):
    category: ExpenseCategory
    limit_amount: float

class ExpenseResponse(BaseModel):
    id: int
    category: str
    amount: float
    description: str
    date: str
    timestamp: str

# Authentication helper functions
def verify_password(plain_password, hashed_password):
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> int:
    """Get current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
        user_id = int(user_id_str)
    except (JWTError, ValueError):
        raise credentials_exception
    
    # Verify user exists
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            if user is None:
                raise credentials_exception
    except Exception as e:
        logger.error(f"Error validating user: {e}")
        raise credentials_exception
    
    return user_id

# Helper functions
def get_user_expenses(user_id: int, days: int = 30):
    """Get user expenses from last N days"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            start_date = datetime.now() - timedelta(days=days)
            cursor.execute('''
                SELECT * FROM expenses 
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (user_id, start_date))
            expenses = cursor.fetchall()
            return expenses
    except Exception as e:
        logger.error(f"Error fetching expenses for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching expenses")

def get_spending_by_category(user_id: int, days: int = 30):
    """Get spending breakdown by category"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            start_date = datetime.now() - timedelta(days=days)
            cursor.execute('''
                SELECT category, SUM(amount) as total
                FROM expenses
                WHERE user_id = ? AND timestamp >= ?
                GROUP BY category
                ORDER BY total DESC
            ''', (user_id, start_date))
            breakdown = cursor.fetchall()
            return breakdown
    except Exception as e:
        logger.error(f"Error fetching spending breakdown for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching spending breakdown")

def get_budget_recommendations(user_id: int):
    """Generate AI-powered budget recommendations"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get user
            cursor.execute('SELECT monthly_budget FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            if not user:
                return None
            
            monthly_budget = user[0]
            
            # Get spending by category (last 3 months)
            spending = get_spending_by_category(user_id, days=90)
            
            recommendations = {
                'total_budget': monthly_budget,
                'recommended_allocation': {
                    'Food': monthly_budget * 0.30,
                    'Transport': monthly_budget * 0.15,
                    'Utilities': monthly_budget * 0.15,
                    'Entertainment': monthly_budget * 0.10,
                    'Shopping': monthly_budget * 0.15,
                    'Health': monthly_budget * 0.10,
                    'Education': monthly_budget * 0.05
                },
                'current_spending': {item[0]: item[1] for item in spending},
                'tips': []
            }
            
            # Generate tips based on spending
            total_spent = sum([item[1] for item in spending])
            
            if total_spent > monthly_budget:
                recommendations['tips'].append(
                    f"⚠️ You're spending ₹{total_spent - monthly_budget:.0f} more than your monthly budget!"
                )
            
            for category, amount in spending:
                recommended = recommendations['recommended_allocation'].get(category, 0)
                if amount > recommended * 1.2:
                    recommendations['tips'].append(
                        f"💡 Consider reducing spending on {category}. Current: ₹{amount:.0f}, Recommended: ₹{recommended:.0f}"
                    )
            
            return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating recommendations")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Smart Expense Tracker & Budget Optimizer API",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/api/auth",
            "users": "/api/users",
            "expenses": "/api/expenses",
            "analytics": "/api/analytics",
            "recommendations": "/api/recommendations",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/api/auth/register", response_model=UserResponse)
@limiter.limit("5/minute")
async def register(request: Request, user: UserRegister):
    """Register a new user"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Hash password
            password_hash = get_password_hash(user.password)
            
            cursor.execute(
                'INSERT INTO users (email, name, monthly_budget, password_hash) VALUES (?, ?, ?, ?)',
                (user.email, user.name, user.monthly_budget, password_hash)
            )
            user_id = cursor.lastrowid
            
            logger.info(f"New user registered: {user.email}")
            
            return UserResponse(
                id=user_id,
                email=user.email,
                name=user.name,
                monthly_budget=user.monthly_budget
            )
    except sqlite3.IntegrityError:
        logger.warning(f"Registration attempt with existing email: {user.email}")
        raise HTTPException(status_code=400, detail="User already exists")
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail="Error creating user")

@app.post("/api/auth/login", response_model=Token)
@limiter.limit("10/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """Login user and return JWT token"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, password_hash FROM users WHERE email = ?', (form_data.username,))
            user = cursor.fetchone()
            
            if not user:
                logger.warning(f"Login attempt with non-existent email: {form_data.username}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect email or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            user_id = user[0]
            password_hash = user[1]
            
            if not verify_password(form_data.password, password_hash):
                logger.warning(f"Failed login attempt for email: {form_data.username}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect email or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": str(user_id)}, expires_delta=access_token_expires
            )
            
            logger.info(f"User logged in: {form_data.username}")
            
            return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(status_code=500, detail="Error during login")

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user_id: int = Depends(get_current_user)):
    """Get current authenticated user information"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, email, name, monthly_budget FROM users WHERE id = ?', (current_user_id,))
            user = cursor.fetchone()
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            return UserResponse(id=user[0], email=user[1], name=user[2], monthly_budget=user[3])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user info: {e}")
        raise HTTPException(status_code=500, detail="Error fetching user information")

@app.post("/api/users")
@limiter.limit("5/minute")
async def create_user(request: Request, user: User):
    """Create a new user (deprecated - use /api/auth/register instead)"""
    logger.warning("Deprecated endpoint /api/users called - use /api/auth/register instead")
    raise HTTPException(
        status_code=400, 
        detail="This endpoint is deprecated. Please use /api/auth/register with password"
    )

@app.get("/api/users/{user_id}")
async def get_user(user_id: int, current_user_id: int = Depends(get_current_user)):
    """Get user profile (requires authentication)"""
    # Verify user can only access their own profile
    if user_id != current_user_id:
        logger.warning(f"User {current_user_id} attempted to access user {user_id} profile")
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, email, name, monthly_budget, created_at FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            return {
                'id': user[0],
                'email': user[1],
                'name': user[2],
                'monthly_budget': user[3],
                'created_at': user[4]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching user")

@app.post("/api/expenses")
@limiter.limit("20/minute")
async def add_expense(request: Request, expense: Expense, current_user_id: int = Depends(get_current_user)):
    """Add a new expense (requires authentication)"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            date = expense.date or datetime.now().strftime('%Y-%m-%d')
            
            cursor.execute(
                'INSERT INTO expenses (user_id, category, amount, description, date) VALUES (?, ?, ?, ?, ?)',
                (current_user_id, expense.category.value, expense.amount, expense.description, date)
            )
            
            expense_id = cursor.lastrowid
            
            logger.info(f"Expense added for user {current_user_id}: {expense.category.value} - ₹{expense.amount}")
            
            return {
                'id': expense_id,
                'message': 'Expense added successfully'
            }
    except Exception as e:
        logger.error(f"Error adding expense for user {current_user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error adding expense")

@app.get("/api/expenses/{user_id}")
async def get_expenses(user_id: int, days: int = 30, current_user_id: int = Depends(get_current_user)):
    """Get user expenses (requires authentication)"""
    # Verify user can only access their own expenses
    if user_id != current_user_id:
        logger.warning(f"User {current_user_id} attempted to access expenses of user {user_id}")
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Verify user exists
            cursor.execute('SELECT id FROM users WHERE id = ?', (user_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="User not found")
            
            start_date = datetime.now() - timedelta(days=days)
            cursor.execute('''
                SELECT id, category, amount, description, date, timestamp
                FROM expenses 
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (user_id, start_date))
            
            expenses = [{
                'id': row[0],
                'category': row[1],
                'amount': row[2],
                'description': row[3],
                'date': row[4],
                'timestamp': row[5]
            } for row in cursor.fetchall()]
            
            return {
                'user_id': user_id,
                'total_expenses': len(expenses),
                'total_spent': sum([e['amount'] for e in expenses]),
                'expenses': expenses
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching expenses for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching expenses")

@app.get("/api/analytics/{user_id}")
async def get_analytics(user_id: int, days: int = 30, current_user_id: int = Depends(get_current_user)):
    """Get spending analytics (requires authentication)"""
    # Verify user can only access their own analytics
    if user_id != current_user_id:
        logger.warning(f"User {current_user_id} attempted to access analytics of user {user_id}")
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Verify user exists
            cursor.execute('SELECT monthly_budget FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            monthly_budget = user[0]
            
            # Spending by category
            start_date = datetime.now() - timedelta(days=days)
            cursor.execute('''
                SELECT category, SUM(amount) as total, COUNT(*) as count
                FROM expenses
                WHERE user_id = ? AND timestamp >= ?
                GROUP BY category
                ORDER BY total DESC
            ''', (user_id, start_date))
            
            breakdown = [{
                'category': row[0],
                'total': row[1],
                'count': row[2]
            } for row in cursor.fetchall()]
            
            # Daily spending
            cursor.execute('''
                SELECT DATE(date), SUM(amount)
                FROM expenses
                WHERE user_id = ? AND timestamp >= ?
                GROUP BY DATE(date)
                ORDER BY DATE(date)
            ''', (user_id, start_date))
            
            daily_spending = [{
                'date': row[0],
                'amount': row[1]
            } for row in cursor.fetchall()]
            
            total_spent = sum([b['total'] for b in breakdown])
            avg_daily = total_spent / days if days > 0 else 0
            
            return {
                'user_id': user_id,
                'period_days': days,
                'total_spent': total_spent,
                'average_daily': avg_daily,
                'monthly_budget': monthly_budget,
                'budget_remaining': monthly_budget - (total_spent / (days / 30)),
                'by_category': breakdown,
                'daily_breakdown': daily_spending
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching analytics for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching analytics")

@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: int, current_user_id: int = Depends(get_current_user)):
    """Get AI-powered budget recommendations (requires authentication)"""
    # Verify user can only access their own recommendations
    if user_id != current_user_id:
        logger.warning(f"User {current_user_id} attempted to access recommendations of user {user_id}")
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    try:
        recommendations = get_budget_recommendations(user_id)
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="User not found")
        
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching recommendations")

if __name__ == '__main__':
    import uvicorn
    import os
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)

