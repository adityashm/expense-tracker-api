from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator
from datetime import datetime, timedelta, timezone
import sqlite3
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import os
from contextlib import contextmanager
from dotenv import load_dotenv
from passlib.context import CryptContext
from jose import JWTError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
DATABASE_URL = os.getenv("DATABASE_URL", "expenses.db")
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = "dev-secret-key-change-in-production"
    logger.warning("SECRET_KEY not set in environment! Using default key - DO NOT USE IN PRODUCTION!")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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

# CORS configuration - restrict to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Database context manager
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()

# Database initialization
def init_db():
    """Initialize database with tables and indexes"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    password TEXT NOT NULL,
                    monthly_budget REAL DEFAULT 50000,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    amount REAL NOT NULL,
                    description TEXT,
                    date DATE DEFAULT CURRENT_DATE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS budgets (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    limit_amount REAL NOT NULL,
                    month INTEGER,
                    year INTEGER,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_expenses_user_id 
                ON expenses(user_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_expenses_timestamp 
                ON expenses(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_expenses_category 
                ON expenses(category)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_budgets_user_id 
                ON budgets(user_id)
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
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
class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str
    monthly_budget: float = 50000
    
    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError('name cannot be empty')
        if len(v) > 100:
            raise ValueError('name must be less than 100 characters')
        return v.strip()
    
    @field_validator('password')
    @classmethod
    def password_must_be_strong(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError('password must be at least 6 characters')
        if len(v) > 100:
            raise ValueError('password must be less than 100 characters')
        return v
    
    @field_validator('monthly_budget')
    @classmethod
    def budget_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('monthly budget must be positive')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    monthly_budget: Optional[float] = None
    
    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if not v or len(v.strip()) == 0:
                raise ValueError('name cannot be empty')
            if len(v) > 100:
                raise ValueError('name must be less than 100 characters')
            return v.strip()
        return v
    
    @field_validator('monthly_budget')
    @classmethod
    def budget_must_be_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError('monthly budget must be positive')
        return v

class User(BaseModel):
    email: str
    name: str
    monthly_budget: float = 50000

class Expense(BaseModel):
    user_id: int
    category: ExpenseCategory
    amount: float
    description: str = ""
    date: Optional[str] = None
    
    @field_validator('user_id')
    @classmethod
    def user_id_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('user_id must be positive')
        return v
    
    @field_validator('amount')
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('amount must be positive')
        return v
    
    @field_validator('description')
    @classmethod
    def description_length(cls, v: str) -> str:
        if len(v) > 500:
            raise ValueError('description must be less than 500 characters')
        return v
    
    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('date must be in YYYY-MM-DD format')
        return v

class ExpenseUpdate(BaseModel):
    category: Optional[ExpenseCategory] = None
    amount: Optional[float] = None
    description: Optional[str] = None
    date: Optional[str] = None
    
    @field_validator('amount')
    @classmethod
    def amount_must_be_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError('amount must be positive')
        return v
    
    @field_validator('description')
    @classmethod
    def description_length(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > 500:
            raise ValueError('description must be less than 500 characters')
        return v
    
    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('date must be in YYYY-MM-DD format')
        return v

class BudgetSet(BaseModel):
    user_id: int
    category: ExpenseCategory
    limit_amount: float
    
    @field_validator('user_id')
    @classmethod
    def user_id_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('user_id must be positive')
        return v
    
    @field_validator('limit_amount')
    @classmethod
    def limit_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('limit amount must be positive')
        return v

class ExpenseResponse(BaseModel):
    id: int
    category: str
    amount: float
    description: str
    date: str
    timestamp: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None

# Helper functions
# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
        return {"id": int(user_id)}
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get the current authenticated user from JWT token"""
    token = credentials.credentials
    user = verify_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify user still exists in database
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, email, name FROM users WHERE id = ?', (user['id'],))
            db_user = cursor.fetchone()
            if not db_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return {"id": db_user[0], "email": db_user[1], "name": db_user[2]}
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error validating user"
        )

def get_user_expenses(user_id: int, days: int = 30):
    """Get user expenses from last N days"""
    try:
        with get_db_connection() as conn:
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
        logger.error(f"Error fetching user expenses: {e}")
        raise

def get_spending_by_category(user_id: int, days: int = 30):
    """Get spending breakdown by category"""
    try:
        with get_db_connection() as conn:
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
        logger.error(f"Error fetching spending by category: {e}")
        raise

def get_budget_recommendations(user_id: int):
    """Generate AI-powered budget recommendations"""
    try:
        with get_db_connection() as conn:
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
                'current_spending': dict(spending),
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
        logger.error(f"Error generating budget recommendations: {e}")
        raise

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Expense Tracker & Budget Optimizer API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "register": "/api/auth/register",
            "login": "/api/auth/login",
            "users": "/api/users",
            "expenses": "/api/expenses",
            "analytics": "/api/analytics",
            "recommendations": "/api/recommendations"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check database connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            cursor.fetchone()
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def register(request: Request, user: UserCreate):
    """Register a new user"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Hash password
            hashed_password = get_password_hash(user.password)
            
            cursor.execute(
                'INSERT INTO users (email, name, password, monthly_budget) VALUES (?, ?, ?, ?)',
                (user.email, user.name, hashed_password, user.monthly_budget)
            )
            conn.commit()
            user_id = cursor.lastrowid
            
            logger.info(f"New user registered: {user.email}")
            
            return {
                'id': user_id,
                'email': user.email,
                'name': user.name,
                'monthly_budget': user.monthly_budget,
                'message': 'User registered successfully'
            }
    except sqlite3.IntegrityError:
        logger.warning(f"Registration attempt with existing email: {user.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating user"
        )

@app.post("/api/auth/login")
@limiter.limit("10/minute")
async def login(request: Request, user_login: UserLogin):
    """Login and get JWT token"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT id, email, name, password FROM users WHERE email = ?',
                (user_login.email,)
            )
            user = cursor.fetchone()
            
            if not user or not verify_password(user_login.password, user[3]):
                logger.warning(f"Failed login attempt for: {user_login.email}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect email or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Create access token
            access_token = create_access_token(
                data={"sub": str(user[0])}
            )
            
            logger.info(f"User logged in: {user_login.email}")
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": user[0],
                    "email": user[1],
                    "name": user[2]
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during login"
        )

@app.post("/api/users")
async def create_user(user: User):
    """Create a new user (deprecated - use /api/auth/register instead)"""
    logger.warning("Deprecated endpoint /api/users called, redirecting to /api/auth/register")
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="This endpoint is deprecated. Please use /api/auth/register instead"
    )

@app.get("/api/users/{user_id}")
async def get_user(user_id: int, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get user profile (protected)"""
    # Users can only access their own profile
    if current_user['id'] != user_id:
        logger.warning(f"User {current_user['id']} attempted to access user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden - you can only access your own profile"
        )
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT id, email, name, monthly_budget, created_at FROM users WHERE id = ?',
                (user_id,)
            )
            user = cursor.fetchone()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
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
        logger.error(f"Error fetching user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching user profile"
        )

@app.put("/api/users/{user_id}")
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update user profile (protected)"""
    # Users can only update their own profile
    if current_user['id'] != user_id:
        logger.warning(f"User {current_user['id']} attempted to update user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden - you can only update your own profile"
        )
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build update query dynamically based on provided fields
            # Using whitelist approach for security
            allowed_fields = {"name": "name = ?", "monthly_budget": "monthly_budget = ?"}
            update_fields = []
            values = []
            
            if user_update.name is not None:
                update_fields.append(allowed_fields["name"])
                values.append(user_update.name)
            
            if user_update.monthly_budget is not None:
                update_fields.append(allowed_fields["monthly_budget"])
                values.append(user_update.monthly_budget)
            
            if not update_fields:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No fields to update"
                )
            
            values.append(user_id)
            query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
            
            cursor.execute(query, values)
            conn.commit()
            
            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            logger.info(f"User {user_id} updated profile")
            
            return {"message": "User updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user profile"
        )

@app.post("/api/expenses", status_code=status.HTTP_201_CREATED)
@limiter.limit("30/minute")
async def add_expense(
    request: Request,
    expense: Expense,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Add a new expense (protected)"""
    # Users can only add expenses for themselves
    if current_user['id'] != expense.user_id:
        logger.warning(f"User {current_user['id']} attempted to add expense for user {expense.user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden - you can only add expenses for yourself"
        )
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Verify user exists
            cursor.execute('SELECT id FROM users WHERE id = ?', (expense.user_id,))
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            date = expense.date or datetime.now().strftime('%Y-%m-%d')
            
            cursor.execute(
                'INSERT INTO expenses (user_id, category, amount, description, date) VALUES (?, ?, ?, ?, ?)',
                (expense.user_id, expense.category.value, expense.amount, expense.description, date)
            )
            
            conn.commit()
            expense_id = cursor.lastrowid
            
            logger.info(f"User {current_user['id']} added expense {expense_id}")
            
            return {
                'id': expense_id,
                'message': 'Expense added successfully'
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding expense: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error adding expense"
        )

@app.get("/api/expenses/{user_id}")
async def get_expenses(
    user_id: int,
    days: int = 30,
    page: int = 1,
    per_page: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get user expenses with pagination (protected)"""
    # Users can only access their own expenses
    if current_user['id'] != user_id:
        logger.warning(f"User {current_user['id']} attempted to access expenses of user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden - you can only access your own expenses"
        )
    
    # Validate pagination parameters
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page must be >= 1"
        )
    if per_page < 1 or per_page > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Per page must be between 1 and 100"
        )
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Verify user exists
            cursor.execute('SELECT id FROM users WHERE id = ?', (user_id,))
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            start_date = datetime.now() - timedelta(days=days)
            offset = (page - 1) * per_page
            
            # Get total count
            cursor.execute('''
                SELECT COUNT(*)
                FROM expenses 
                WHERE user_id = ? AND timestamp >= ?
            ''', (user_id, start_date))
            total_count = cursor.fetchone()[0]
            
            # Get paginated expenses
            cursor.execute('''
                SELECT id, category, amount, description, date, timestamp
                FROM expenses 
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            ''', (user_id, start_date, per_page, offset))
            
            expenses = [{
                'id': row[0],
                'category': row[1],
                'amount': row[2],
                'description': row[3],
                'date': row[4],
                'timestamp': row[5]
            } for row in cursor.fetchall()]
            
            total_pages = (total_count + per_page - 1) // per_page
            
            return {
                'user_id': user_id,
                'total_expenses': total_count,
                'total_spent': sum([e['amount'] for e in expenses]),
                'expenses': expenses,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total_pages': total_pages,
                    'total_count': total_count
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching expenses: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching expenses"
        )

@app.put("/api/expenses/{expense_id}")
async def update_expense(
    expense_id: int,
    expense_update: ExpenseUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update an expense (protected)"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Verify expense exists and belongs to current user
            cursor.execute('SELECT user_id FROM expenses WHERE id = ?', (expense_id,))
            expense = cursor.fetchone()
            
            if not expense:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Expense not found"
                )
            
            if expense[0] != current_user['id']:
                logger.warning(f"User {current_user['id']} attempted to update expense {expense_id} owned by user {expense[0]}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access forbidden - you can only update your own expenses"
                )
            
            # Build update query dynamically using whitelist approach for security
            allowed_fields = {
                "category": "category = ?",
                "amount": "amount = ?",
                "description": "description = ?",
                "date": "date = ?"
            }
            update_fields = []
            values = []
            
            if expense_update.category is not None:
                update_fields.append(allowed_fields["category"])
                values.append(expense_update.category.value)
            
            if expense_update.amount is not None:
                update_fields.append(allowed_fields["amount"])
                values.append(expense_update.amount)
            
            if expense_update.description is not None:
                update_fields.append(allowed_fields["description"])
                values.append(expense_update.description)
            
            if expense_update.date is not None:
                update_fields.append(allowed_fields["date"])
                values.append(expense_update.date)
            
            if not update_fields:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No fields to update"
                )
            
            values.append(expense_id)
            query = f"UPDATE expenses SET {', '.join(update_fields)} WHERE id = ?"
            
            cursor.execute(query, values)
            conn.commit()
            
            logger.info(f"User {current_user['id']} updated expense {expense_id}")
            
            return {"message": "Expense updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating expense: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating expense"
        )

@app.delete("/api/expenses/{expense_id}", status_code=status.HTTP_200_OK)
async def delete_expense(
    expense_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete an expense (protected)"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Verify expense exists and belongs to current user
            cursor.execute('SELECT user_id FROM expenses WHERE id = ?', (expense_id,))
            expense = cursor.fetchone()
            
            if not expense:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Expense not found"
                )
            
            if expense[0] != current_user['id']:
                logger.warning(f"User {current_user['id']} attempted to delete expense {expense_id} owned by user {expense[0]}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access forbidden - you can only delete your own expenses"
                )
            
            cursor.execute('DELETE FROM expenses WHERE id = ?', (expense_id,))
            conn.commit()
            
            logger.info(f"User {current_user['id']} deleted expense {expense_id}")
            
            return {"message": "Expense deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting expense: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting expense"
        )

@app.get("/api/analytics/{user_id}")
async def get_analytics(
    user_id: int,
    days: int = 30,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get spending analytics (protected)"""
    # Users can only access their own analytics
    if current_user['id'] != user_id:
        logger.warning(f"User {current_user['id']} attempted to access analytics of user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden - you can only access your own analytics"
        )
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Verify user exists
            cursor.execute('SELECT monthly_budget FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
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
            
            # Calculate budget remaining - handle edge cases
            if days > 0:
                budget_remaining = user[0] - (total_spent / (days / 30))
            else:
                budget_remaining = user[0]
            
            return {
                'user_id': user_id,
                'period_days': days,
                'total_spent': total_spent,
                'average_daily': avg_daily,
                'monthly_budget': user[0],
                'budget_remaining': budget_remaining,
                'by_category': breakdown,
                'daily_breakdown': daily_spending
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching analytics"
        )

@app.get("/api/recommendations/{user_id}")
async def get_recommendations(
    user_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get AI-powered budget recommendations (protected)"""
    # Users can only access their own recommendations
    if current_user['id'] != user_id:
        logger.warning(f"User {current_user['id']} attempted to access recommendations of user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden - you can only access your own recommendations"
        )
    
    try:
        recommendations = get_budget_recommendations(user_id)
        
        if not recommendations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching recommendations"
        )

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
