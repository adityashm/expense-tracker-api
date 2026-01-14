from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import sqlite3
from typing import List, Optional
from enum import Enum
import json

app = FastAPI(
    title="Smart Expense Tracker API",
    description="Track expenses and get AI-powered budget recommendations",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database initialization
def init_db():
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE,
            name TEXT,
            monthly_budget REAL DEFAULT 50000,
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
    
    conn.commit()
    conn.close()

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

class BudgetSet(BaseModel):
    user_id: int
    category: ExpenseCategory
    limit_amount: float

class ExpenseResponse(BaseModel):
    id: int
    category: str
    amount: float
    description: str
    date: str
    timestamp: str

# Helper functions
def get_user_expenses(user_id: int, days: int = 30):
    """Get user expenses from last N days"""
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    
    start_date = datetime.now() - timedelta(days=days)
    cursor.execute('''
        SELECT * FROM expenses 
        WHERE user_id = ? AND timestamp >= ?
        ORDER BY timestamp DESC
    ''', (user_id, start_date))
    
    expenses = cursor.fetchall()
    conn.close()
    return expenses

def get_spending_by_category(user_id: int, days: int = 30):
    """Get spending breakdown by category"""
    conn = sqlite3.connect('expenses.db')
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
    conn.close()
    return breakdown

def get_budget_recommendations(user_id: int):
    """Generate AI-powered budget recommendations"""
    conn = sqlite3.connect('expenses.db')
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
    
    conn.close()
    return recommendations

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Smart Expense Tracker & Budget Optimizer API",
        "version": "1.0.0",
        "endpoints": {
            "users": "/api/users",
            "expenses": "/api/expenses",
            "analytics": "/api/analytics",
            "recommendations": "/api/recommendations"
        }
    }

@app.post("/api/users")
async def create_user(user: User):
    """Create a new user"""
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            'INSERT INTO users (email, name, monthly_budget) VALUES (?, ?, ?)',
            (user.email, user.name, user.monthly_budget)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        return {
            'id': user_id,
            'email': user.email,
            'name': user.name,
            'monthly_budget': user.monthly_budget,
            'message': 'User created successfully'
        }
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="User already exists")

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    """Get user profile"""
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, email, name, monthly_budget, created_at FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        'id': user[0],
        'email': user[1],
        'name': user[2],
        'monthly_budget': user[3],
        'created_at': user[4]
    }

@app.post("/api/expenses")
async def add_expense(expense: Expense):
    """Add a new expense"""
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    
    # Verify user exists
    cursor.execute('SELECT id FROM users WHERE id = ?', (expense.user_id,))
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="User not found")
    
    date = expense.date or datetime.now().strftime('%Y-%m-%d')
    
    cursor.execute(
        'INSERT INTO expenses (user_id, category, amount, description, date) VALUES (?, ?, ?, ?, ?)',
        (expense.user_id, expense.category.value, expense.amount, expense.description, date)
    )
    
    conn.commit()
    expense_id = cursor.lastrowid
    conn.close()
    
    return {
        'id': expense_id,
        'message': 'Expense added successfully'
    }

@app.get("/api/expenses/{user_id}")
async def get_expenses(user_id: int, days: int = 30):
    """Get user expenses"""
    conn = sqlite3.connect('expenses.db')
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
    
    conn.close()
    
    return {
        'user_id': user_id,
        'total_expenses': len(expenses),
        'total_spent': sum([e['amount'] for e in expenses]),
        'expenses': expenses
    }

@app.get("/api/analytics/{user_id}")
async def get_analytics(user_id: int, days: int = 30):
    """Get spending analytics"""
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    
    # Verify user exists
    cursor.execute('SELECT monthly_budget FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
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
    
    conn.close()
    
    total_spent = sum([b['total'] for b in breakdown])
    avg_daily = total_spent / days if days > 0 else 0
    
    return {
        'user_id': user_id,
        'period_days': days,
        'total_spent': total_spent,
        'average_daily': avg_daily,
        'monthly_budget': user[0],
        'budget_remaining': user[0] - (total_spent / (days / 30)),
        'by_category': breakdown,
        'daily_breakdown': daily_spending
    }

@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: int):
    """Get AI-powered budget recommendations"""
    recommendations = get_budget_recommendations(user_id)
    
    if not recommendations:
        raise HTTPException(status_code=404, detail="User not found")
    
    return recommendations

if __name__ == '__main__':
    import uvicorn
    import os
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)

