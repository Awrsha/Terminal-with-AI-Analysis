from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from werkzeug.middleware.proxy_fix import ProxyFix
from sqlalchemy.sql import func
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import sys
import json
import time
import uuid
import re
import secrets
import subprocess
import threading
import traceback
import logging
import requests
import hashlib
import shutil
import tempfile
import platform
import socket
import psutil
import signal
import base64
import mimetypes
import zipfile
import io
from pathlib import Path
from urllib.parse import urlparse
from functools import wraps
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from collections import defaultdict, deque, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Basic Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    TESTING = os.environ.get('FLASK_TESTING', 'False').lower() == 'true'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///terminal.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = DEBUG
    
    # API Keys
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY') or "token here"
    HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN') or "token here"
    
    # Command execution settings
    CMD_EXECUTION_TIMEOUT = int(os.environ.get('CMD_EXECUTION_TIMEOUT') or 30)  # seconds
    MAX_OUTPUT_SIZE = int(os.environ.get('MAX_OUTPUT_SIZE') or 1024 * 100)  # 100KB
    
    # Security settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    RATE_LIMIT_DEFAULT = os.environ.get('RATE_LIMIT_DEFAULT', '100 per day, 10 per minute')
    BLOCKED_COMMANDS = [
        'format', 'del', 'rmdir', 'rm', 'shutdown', 'taskkill', 'net user',
        'cacls', 'attrib', 'diskpart', 'cipher', 'takeown', 'icacls'
    ]
    
    # Cache settings
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'SimpleCache')
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT') or 300)
    
    # File upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or os.path.join(os.getcwd(), 'uploads')
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH') or 16 * 1024 * 1024)  # 16MB
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(days=int(os.environ.get('SESSION_LIFETIME_DAYS') or 1))
    SESSION_TYPE = os.environ.get('SESSION_TYPE', 'filesystem')
    SESSION_FILE_DIR = os.environ.get('SESSION_FILE_DIR', './flask_session')
    SESSION_PERMANENT = True
    SESSION_USE_SIGNER = True
    
    # AI Analysis Settings
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL = os.environ.get('GROQ_MODEL', "llama3-70b-8192")
    AI_ANALYSIS_MAX_TOKENS = int(os.environ.get('AI_ANALYSIS_MAX_TOKENS') or 1000)
    AI_ANALYSIS_TEMPERATURE = float(os.environ.get('AI_ANALYSIS_TEMPERATURE') or 0.7)
    
    # Terminal settings
    TERMINAL_HISTORY_LIMIT = int(os.environ.get('TERMINAL_HISTORY_LIMIT') or 100)  # commands per user
    
    # Admin settings
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin')
    ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'admin@example.com')
    
    # Environment-specific configurations
    @classmethod
    def init_env_configs(cls):
        env = os.environ.get('FLASK_ENV', 'development')
        if env == 'production':
            cls.SESSION_COOKIE_SECURE = True
            cls.SESSION_COOKIE_HTTPONLY = True
            cls.REMEMBER_COOKIE_SECURE = True
            cls.REMEMBER_COOKIE_HTTPONLY = True
            cls.DEBUG = False
        return cls

# Initialize the Flask application
app = Flask(__name__, 
    static_folder='static',
    template_folder='templates',
    instance_relative_config=True
)

# Load configuration
app.config.from_object(Config.init_env_configs())

# Configure Flask to trust proxies if behind a reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Configure CORS
CORS(app, resources={r"/api/*": {"origins": app.config['CORS_ORIGINS']}})

# Configure rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[app.config['RATE_LIMIT_DEFAULT']],
    storage_uri="memory://",
)

# Configure cache
cache = Cache(config={'CACHE_TYPE': app.config['CACHE_TYPE']})
cache.init_app(app)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), default='user')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    commands = db.relationship('Command', backref='user', lazy='dynamic')
    api_keys = db.relationship('ApiKey', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        return self.role == 'admin'

class Command(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    command_text = db.Column(db.String(1000), nullable=False)
    output = db.Column(db.Text, nullable=True)
    exit_code = db.Column(db.Integer, nullable=True)
    execution_time = db.Column(db.Float, nullable=True)  # in seconds
    analysis = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.String(36), nullable=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'command': self.command_text,
            'output': self.output,
            'exit_code': self.exit_code,
            'execution_time': self.execution_time,
            'analysis': self.analysis,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'username': self.user.username,
        }

class ApiKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used = db.Column(db.DateTime, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __init__(self, name, user_id):
        self.name = name
        self.user_id = user_id
        self.key = self._generate_key()
    
    def _generate_key(self):
        return f"tk_{secrets.token_hex(32)}"
    
    def update_last_used(self):
        self.last_used = datetime.utcnow()
        db.session.commit()

class SystemMetric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cpu_usage = db.Column(db.Float, nullable=True)
    memory_usage = db.Column(db.Float, nullable=True)
    disk_usage = db.Column(db.Float, nullable=True)
    command_count = db.Column(db.Integer, default=0)
    error_count = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    @classmethod
    def record_metrics(cls):
        try:
            # Get system metrics
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            
            # Get command and error counts in the last hour
            past_hour = datetime.utcnow() - timedelta(hours=1)
            command_count = Command.query.filter(Command.created_at >= past_hour).count()
            error_count = Command.query.filter(
                Command.created_at >= past_hour,
                Command.exit_code != 0
            ).count()
            
            # Create and save metrics
            metric = cls(
                cpu_usage=cpu,
                memory_usage=memory,
                disk_usage=disk,
                command_count=command_count,
                error_count=error_count
            )
            db.session.add(metric)
            db.session.commit()
            return metric
        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")
            return None

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    details = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    @classmethod
    def log_event(cls, event_type, user_id=None, ip_address=None, details=None):
        try:
            log = cls(
                event_type=event_type,
                user_id=user_id,
                ip_address=ip_address,
                details=details
            )
            db.session.add(log)
            db.session.commit()
            return log
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            return None

class BlockedIP(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(50), unique=True, nullable=False)
    reason = db.Column(db.String(255), nullable=True)
    blocked_at = db.Column(db.DateTime, default=datetime.utcnow)
    blocked_until = db.Column(db.DateTime, nullable=True)
    
    @classmethod
    def is_blocked(cls, ip_address):
        blocked = cls.query.filter_by(ip_address=ip_address).first()
        if not blocked:
            return False
        
        if blocked.blocked_until and blocked.blocked_until < datetime.utcnow():
            db.session.delete(blocked)
            db.session.commit()
            return False
        
        return True

# Login Manager setup
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Utility Functions
class CommandExecutor:
    def __init__(self, command: str, timeout: int = None, cwd: str = None):
        self.command = command
        self.timeout = timeout or app.config['CMD_EXECUTION_TIMEOUT']
        self.cwd = cwd or os.getcwd()
        self.process = None
        self.output = None
        self.exit_code = None
        self.error = None
        self.execution_time = None
        self._persistent_dir = None
    
    def is_dangerous_command(self) -> bool:
        """Check if the command is potentially dangerous."""
        cmd_lower = self.command.lower()
        
        # Check against blocked commands
        for blocked in app.config['BLOCKED_COMMANDS']:
            if blocked.lower() in cmd_lower:
                return True
        
        # Regular expressions to detect potentially dangerous patterns
        dangerous_patterns = [
            r'rm\s+(-rf?|/s)\s+[/\\]',  # Recursive delete from root
            r'>(>?)\s*/dev/(null|zero)',  # Redirecting to special devices
            r'mkfs',  # Formatting drives
            r'dd\s+if=.*\s+of=',  # Direct disk operations
            r'chmod\s+777',  # Overly permissive permissions
            r'^sudo\s+',  # Commands with sudo
            r'passwd',  # Password changing
            r'useradd|userdel',  # User management
            r'(wget|curl)\s+.*\s*\|\s*bash',  # Piping downloaded content to bash
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, cmd_lower):
                return True
        
        return False
    
    def execute(self) -> Dict[str, Any]:
        """Execute the command and return the result."""
        if self.is_dangerous_command():
            return {
                'output': "Error: This command has been blocked for security reasons.",
                'exit_code': 1,
                'execution_time': 0,
                'error': "Command blocked for security reasons"
            }
        
        # Special handling for 'cd' command to maintain directory state
        if self.command.lower().startswith('cd '):
            return self._handle_cd_command()
        
        start_time = time.time()
        try:
            # Ensure we're using cmd.exe with full environment 
            # and running command as a single string to preserve special characters
            if platform.system() == 'Windows':
                # For Windows, use explicit PATH to find command executables
                env = os.environ.copy()
                # Add common Windows executable paths if not already in PATH
                system_paths = [
                    r'C:\Windows\System32',
                    r'C:\Windows',
                    r'C:\Windows\System32\Wbem',
                    r'C:\Windows\System32\WindowsPowerShell\v1.0'
                ]
                for path in system_paths:
                    if path not in env['PATH']:
                        env['PATH'] = path + os.pathsep + env['PATH']
                
                self.process = subprocess.Popen(
                    ['cmd.exe', '/c', self.command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.cwd,
                    shell=False,  # More secure to set shell=False
                    env=env
                )
            else:
                # For Linux/Unix, use bash
                self.process = subprocess.Popen(
                    ['/bin/bash', '-c', self.command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.cwd,
                    shell=False  # More secure to set shell=False
                )
            
            try:
                stdout, stderr = self.process.communicate(timeout=self.timeout)
                self.exit_code = self.process.returncode
                output = stdout
                
                # If we have stderr output and the command failed, add it to the output
                if stderr and self.exit_code != 0:
                    output += f"\nError:\n{stderr}"
                
                # Handle empty output by returning a message
                if not output.strip():
                    output = "Command executed successfully with no output."
                
                # Truncate output if it's too large
                if len(output) > app.config['MAX_OUTPUT_SIZE']:
                    output = output[:app.config['MAX_OUTPUT_SIZE']] + "\n... (output truncated due to size)"
                
                self.output = output
            except subprocess.TimeoutExpired:
                # Kill the process if it times out
                self.process.kill()
                stdout, stderr = self.process.communicate()
                self.exit_code = -1
                self.output = f"Command timed out after {self.timeout} seconds."
                self.error = "Timeout expired"
        except Exception as e:
            self.exit_code = 1
            self.output = f"Error executing command: {str(e)}\n\nMake sure the command exists and is in your system PATH."
            self.error = str(e)
        finally:
            self.execution_time = time.time() - start_time
        
        return {
            'output': self.output,
            'exit_code': self.exit_code,
            'execution_time': self.execution_time,
            'error': self.error
        }
    
    def _handle_cd_command(self) -> Dict[str, Any]:
        """Special handling for the cd command which needs to maintain state"""
        start_time = time.time()
        try:
            # Extract the directory to change to
            parts = self.command.split(' ', 1)
            if len(parts) < 2:
                directory = os.path.expanduser('~')  # Default to home directory
            else:
                directory = parts[1].strip()
            
            # Handle special cases
            if directory == '..':
                new_dir = os.path.dirname(self.cwd)
            elif directory.startswith('.'):
                new_dir = os.path.abspath(os.path.join(self.cwd, directory))
            elif not os.path.isabs(directory):
                new_dir = os.path.abspath(os.path.join(self.cwd, directory))
            else:
                new_dir = os.path.abspath(directory)
            
            # Check if directory exists
            if os.path.exists(new_dir) and os.path.isdir(new_dir):
                # Update the current working directory
                self.cwd = new_dir
                # Store the new directory in app.config for future commands
                app.config['CURRENT_DIRECTORY'] = new_dir
                
                # Return success
                return {
                    'output': f"Changed directory to: {new_dir}",
                    'exit_code': 0,
                    'execution_time': time.time() - start_time,
                    'error': None
                }
            else:
                return {
                    'output': f"Error: The directory '{directory}' does not exist.",
                    'exit_code': 1,
                    'execution_time': time.time() - start_time,
                    'error': "Directory not found"
                }
        except Exception as e:
            return {
                'output': f"Error changing directory: {str(e)}",
                'exit_code': 1,
                'execution_time': time.time() - start_time,
                'error': str(e)
            }

class AIAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or app.config['GROQ_API_KEY']
        self.api_url = app.config['GROQ_API_URL']
        self.model = app.config['GROQ_MODEL']
        self.max_tokens = app.config['AI_ANALYSIS_MAX_TOKENS']
        self.temperature = app.config['AI_ANALYSIS_TEMPERATURE']
    
    @cache.memoize(timeout=3600)  # Cache for 1 hour
    def analyze_command(self, command: str, output: str) -> str:
        """Analyze the command and its output using GROQ API."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Prepare the messages for the AI
        system_prompt = """
        You are an expert command-line analyst. Your task is to analyze Windows CMD commands and their outputs.
        Provide a detailed but concise analysis explaining:
        1. What the command is doing
        2. Key information in the output
        3. Any notable findings or issues
        4. Potential security implications, if any
        5. Any recommendations or suggestions
        
        Format your analysis in a clear, structured way with appropriate headings.
        Keep your analysis objective, technical, and informative.
        """
        
        user_prompt = f"""
        Command: {command}
        
        Output:
        ```
        {output}
        ```
        
        Please analyze this command and its output.
        """
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract the analysis from the response
            if "choices" in result and len(result["choices"]) > 0:
                analysis = result["choices"][0]["message"]["content"]
                return analysis
            else:
                return "Error: Unable to generate analysis from AI response."
        except requests.RequestException as e:
            logger.error(f"Error calling GROQ API: {e}")
            return f"Error analyzing command: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in AI analysis: {e}")
            return "An unexpected error occurred during analysis."

# Custom decorators
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            return jsonify({"error": "Admin privileges required"}), 403
        return f(*args, **kwargs)
    return decorated_function

def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "API key is required"}), 401
        
        # Check if the API key exists and is active
        key = ApiKey.query.filter_by(key=api_key, is_active=True).first()
        if not key:
            return jsonify({"error": "Invalid or inactive API key"}), 401
        
        # Update last used timestamp
        key.update_last_used()
        
        # Add the user to the request context
        request.user = key.user
        return f(*args, **kwargs)
    return decorated_function

# Middleware to check IP blocklist
@app.before_request
def check_ip_block():
    ip = request.remote_addr
    if ip and BlockedIP.is_blocked(ip):
        return jsonify({'error': 'Your IP address has been blocked'}), 403

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password) and user.is_active:
            login_user(user, remember=request.form.get('remember', False))
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            # Log the login event
            AuditLog.log_event(
                'login',
                user_id=user.id,
                ip_address=request.remote_addr,
                details=f"User {username} logged in"
            )
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            # Log failed login attempt
            AuditLog.log_event(
                'failed_login',
                ip_address=request.remote_addr,
                details=f"Failed login attempt for username: {username}"
            )
            return render_template('login.html', error="Invalid username or password")
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    AuditLog.log_event(
        'logout',
        user_id=current_user.id,
        ip_address=request.remote_addr,
        details=f"User {current_user.username} logged out"
    )
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Basic validation
        if not username or not email or not password:
            return render_template('register.html', error="All fields are required")
        
        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match")
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error="Username already exists")
        
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error="Email already exists")
        
        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        # If this is the first user, make them admin
        if User.query.count() == 0:
            new_user.role = 'admin'
        
        db.session.add(new_user)
        db.session.commit()
        
        # Log the registration event
        AuditLog.log_event(
            'registration',
            user_id=new_user.id,
            ip_address=request.remote_addr,
            details=f"New user registered: {username}"
        )
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/profile')
@login_required
def profile():
    # Get user's command history
    commands = Command.query.filter_by(user_id=current_user.id).order_by(Command.created_at.desc()).limit(20).all()
    api_keys = ApiKey.query.filter_by(user_id=current_user.id).all()
    
    return render_template('profile.html', user=current_user, commands=commands, api_keys=api_keys)

@app.route('/api/keys', methods=['POST'])
@login_required
def create_api_key():
    name = request.json.get('name')
    if not name:
        return jsonify({'error': 'Key name is required'}), 400
    
    api_key = ApiKey(name=name, user_id=current_user.id)
    db.session.add(api_key)
    db.session.commit()
    
    AuditLog.log_event(
        'api_key_created',
        user_id=current_user.id,
        ip_address=request.remote_addr,
        details=f"User {current_user.username} created API key: {name}"
    )
    
    return jsonify({
        'id': api_key.id,
        'key': api_key.key,
        'name': api_key.name,
        'created_at': api_key.created_at.isoformat()
    })

@app.route('/api/keys/<int:key_id>', methods=['DELETE'])
@login_required
def delete_api_key(key_id):
    key = ApiKey.query.get_or_404(key_id)
    
    # Check if the key belongs to the current user
    if key.user_id != current_user.id and not current_user.is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    db.session.delete(key)
    db.session.commit()
    
    AuditLog.log_event(
        'api_key_deleted',
        user_id=current_user.id,
        ip_address=request.remote_addr,
        details=f"User {current_user.username} deleted API key: {key.name}"
    )
    
    return jsonify({'success': True})

# API Routes
@app.route('/api/execute', methods=['POST'])
@limiter.limit("30 per minute")
def execute_command():
    # Get the command from the request
    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({'error': 'Command is required'}), 400
    
    command = data['command'].strip()
    if not command:
        return jsonify({'error': 'Command cannot be empty'}), 400
    
    # Handle special case: 'clear' command
    if command.lower() == 'clear':
        return jsonify({
            'command': command,
            'output': 'Terminal cleared',
            'exit_code': 0,
            'execution_time': 0
        })
    
    # Set up current user or guest user ID
    user_id = current_user.id if current_user.is_authenticated else None
    if user_id is None:
        # Check if API key was provided
        api_key = request.headers.get('X-API-Key')
        if api_key:
            key = ApiKey.query.filter_by(key=api_key, is_active=True).first()
            if key:
                user_id = key.user_id
    
    # If still no user_id, use a guest ID
    if user_id is None:
        # In a real app, you might create a guest user record
        # For this example, we'll use a dummy ID
        user_id = 0  # Guest user ID
    
    # Generate or retrieve session ID from client
    session_id = request.headers.get('X-Session-ID') or str(uuid.uuid4())
    
    # Create a command executor
    executor = CommandExecutor(command)
    result = executor.execute()
    
    # Store the command in the database if we have a valid user
    if user_id:
        cmd = Command(
            command_text=command,
            output=result['output'],
            exit_code=result['exit_code'],
            execution_time=result['execution_time'],
            user_id=user_id,
            session_id=session_id
        )
        db.session.add(cmd)
        db.session.commit()
    
    # Log the command execution
    AuditLog.log_event(
        'command_executed',
        user_id=user_id,
        ip_address=request.remote_addr,
        details=f"Command executed: {command}"
    )
    
    # Return the result to the client
    response = jsonify({
        'command': command,
        'output': result['output'],
        'exit_code': result['exit_code'],
        'execution_time': result['execution_time']
    })
    
    # Set session ID in response header for client to use in future requests
    response.headers['X-Session-ID'] = session_id
    
    return response

@app.route('/api/analyze', methods=['POST'])
@limiter.limit("20 per minute")
def analyze_command():
    # Get the command and output from the request
    data = request.get_json()
    if not data or 'command' not in data or 'output' not in data:
        return jsonify({'error': 'Command and output are required'}), 400
    
    command = data['command'].strip()
    output = data['output'].strip()
    
    if not command or not output:
        return jsonify({'error': 'Command and output cannot be empty'}), 400
    
    # Create an AI analyzer and get the analysis
    analyzer = AIAnalyzer()
    analysis = analyzer.analyze_command(command, output)
    
    # Update the command record with the analysis if user is authenticated
    if current_user.is_authenticated:
        # Get the most recent command with this exact text and output
        cmd = Command.query.filter_by(
            command_text=command,
            output=output,
            user_id=current_user.id
        ).order_by(Command.created_at.desc()).first()
        
        if cmd:
            cmd.analysis = analysis
            db.session.commit()
    
    # Return the analysis to the client
    return jsonify({
        'command': command,
        'analysis': analysis
    })

@app.route('/api/history', methods=['GET'])
@login_required
def get_command_history():
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Limit per_page to a reasonable value
    per_page = min(per_page, 100)
    
    # Get user's command history with pagination
    commands = Command.query.filter_by(user_id=current_user.id).order_by(
        Command.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    result = {
        'commands': [cmd.to_dict() for cmd in commands.items],
        'pagination': {
            'page': commands.page,
            'per_page': commands.per_page,
            'total': commands.total,
            'pages': commands.pages,
            'has_next': commands.has_next,
            'has_prev': commands.has_prev
        }
    }
    
    return jsonify(result)

@app.route('/api/history/<int:command_id>', methods=['GET'])
@login_required
def get_command_detail(command_id):
    # Get the command by ID
    command = Command.query.get_or_404(command_id)
    
    # Check if the command belongs to the current user
    if command.user_id != current_user.id and not current_user.is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    return jsonify(command.to_dict())

@app.route('/api/history/<int:command_id>', methods=['DELETE'])
@login_required
def delete_command(command_id):
    # Get the command by ID
    command = Command.query.get_or_404(command_id)
    
    # Check if the command belongs to the current user
    if command.user_id != current_user.id and not current_user.is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    db.session.delete(command)
    db.session.commit()
    
    AuditLog.log_event(
        'command_deleted',
        user_id=current_user.id,
        ip_address=request.remote_addr,
        details=f"User {current_user.username} deleted command ID: {command_id}"
    )
    
    return jsonify({'success': True})

# Admin routes
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    # Get some basic stats for the dashboard
    total_users = User.query.count()
    total_commands = Command.query.count()
    active_users = db.session.query(func.count(distinct(Command.user_id))).filter(
        Command.created_at >= (datetime.utcnow() - timedelta(days=1))
    ).scalar()
    
    # Get latest system metrics
    latest_metrics = SystemMetric.query.order_by(SystemMetric.timestamp.desc()).first()
    
    # Get recent audit logs
    recent_logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(50).all()
    
    return render_template(
        'admin/dashboard.html',
        total_users=total_users,
        total_commands=total_commands,
        active_users=active_users,
        metrics=latest_metrics,
        logs=recent_logs
    )

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Get users with pagination
    users = User.query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/users.html', users=users)

@app.route('/admin/users/<int:user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_user_edit(user_id):
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        # Update user details
        user.username = request.form.get('username', user.username)
        user.email = request.form.get('email', user.email)
        user.role = request.form.get('role', user.role)
        user.is_active = 'is_active' in request.form
        
        # Update password if provided
        new_password = request.form.get('password')
        if new_password:
            user.set_password(new_password)
        
        db.session.commit()
        
        AuditLog.log_event(
            'user_updated',
            user_id=current_user.id,
            ip_address=request.remote_addr,
            details=f"Admin {current_user.username} updated user: {user.username}"
        )
        
        return redirect(url_for('admin_users'))
    
    # Get user command stats
    command_count = Command.query.filter_by(user_id=user.id).count()
    last_command = Command.query.filter_by(user_id=user.id).order_by(Command.created_at.desc()).first()
    api_key_count = ApiKey.query.filter_by(user_id=user.id).count()
    
    return render_template(
        'admin/user_edit.html',
        user=user,
        command_count=command_count,
        last_command=last_command,
        api_key_count=api_key_count
    )

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_user_delete(user_id):
    user = User.query.get_or_404(user_id)
    
    # Don't allow deleting your own account
    if user.id == current_user.id:
        return jsonify({'error': 'You cannot delete your own account'}), 400
    
    # Store username for logging
    username = user.username
    
    # Delete the user
    db.session.delete(user)
    db.session.commit()
    
    AuditLog.log_event(
        'user_deleted',
        user_id=current_user.id,
        ip_address=request.remote_addr,
        details=f"Admin {current_user.username} deleted user: {username}"
    )
    
    return redirect(url_for('admin_users'))

@app.route('/admin/logs')
@login_required
@admin_required
def admin_logs():
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    event_type = request.args.get('event_type')
    user_id = request.args.get('user_id', type=int)
    
    # Build query with filters
    query = AuditLog.query
    
    if event_type:
        query = query.filter_by(event_type=event_type)
    
    if user_id:
        query = query.filter_by(user_id=user_id)
    
    # Get logs with pagination
    logs = query.order_by(AuditLog.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    # Get unique event types for filter dropdown
    event_types = db.session.query(AuditLog.event_type).distinct().all()
    event_types = [et[0] for et in event_types]
    
    return render_template(
        'admin/logs.html',
        logs=logs,
        event_types=event_types,
        current_event_type=event_type,
        current_user_id=user_id
    )

@app.route('/admin/blocked-ips')
@login_required
@admin_required
def admin_blocked_ips():
    # Get all blocked IPs
    blocked_ips = BlockedIP.query.order_by(BlockedIP.blocked_at.desc()).all()
    
    return render_template('admin/blocked_ips.html', blocked_ips=blocked_ips)

@app.route('/admin/block-ip', methods=['POST'])
@login_required
@admin_required
def admin_block_ip():
    ip_address = request.form.get('ip_address')
    reason = request.form.get('reason')
    duration = request.form.get('duration', type=int)  # Duration in hours
    
    if not ip_address:
        return jsonify({'error': 'IP address is required'}), 400
    
    # Check if IP is already blocked
    existing = BlockedIP.query.filter_by(ip_address=ip_address).first()
    if existing:
        # Update the existing block
        existing.reason = reason
        existing.blocked_at = datetime.utcnow()
        existing.blocked_until = datetime.utcnow() + timedelta(hours=duration) if duration else None
    else:
        # Create a new block
        blocked_ip = BlockedIP(
            ip_address=ip_address,
            reason=reason,
            blocked_until=datetime.utcnow() + timedelta(hours=duration) if duration else None
        )
        db.session.add(blocked_ip)
    
    db.session.commit()
    
    AuditLog.log_event(
        'ip_blocked',
        user_id=current_user.id,
        ip_address=request.remote_addr,
        details=f"Admin {current_user.username} blocked IP: {ip_address}"
    )
    
    return redirect(url_for('admin_blocked_ips'))

@app.route('/admin/unblock-ip/<int:block_id>', methods=['POST'])
@login_required
@admin_required
def admin_unblock_ip(block_id):
    blocked_ip = BlockedIP.query.get_or_404(block_id)
    ip = blocked_ip.ip_address
    
    db.session.delete(blocked_ip)
    db.session.commit()
    
    AuditLog.log_event(
        'ip_unblocked',
        user_id=current_user.id,
        ip_address=request.remote_addr,
        details=f"Admin {current_user.username} unblocked IP: {ip}"
    )
    
    return redirect(url_for('admin_blocked_ips'))

@app.route('/admin/metrics')
@login_required
@admin_required
def admin_metrics():
    # Time range selection
    time_range = request.args.get('range', 'day')
    
    if time_range == 'day':
        since = datetime.utcnow() - timedelta(days=1)
    elif time_range == 'week':
        since = datetime.utcnow() - timedelta(weeks=1)
    elif time_range == 'month':
        since = datetime.utcnow() - timedelta(days=30)
    else:
        since = datetime.utcnow() - timedelta(days=1)
    
    # Get metrics for the selected time range
    metrics = SystemMetric.query.filter(SystemMetric.timestamp >= since).order_by(SystemMetric.timestamp).all()
    
    # Format metrics for charts
    timestamps = [m.timestamp.strftime('%Y-%m-%d %H:%M:%S') for m in metrics]
    cpu_data = [m.cpu_usage for m in metrics]
    memory_data = [m.memory_usage for m in metrics]
    disk_data = [m.disk_usage for m in metrics]
    command_data = [m.command_count for m in metrics]
    error_data = [m.error_count for m in metrics]
    
    return render_template(
        'admin/metrics.html',
        time_range=time_range,
        timestamps=timestamps,
        cpu_data=cpu_data,
        memory_data=memory_data,
        disk_data=disk_data,
        command_data=command_data,
        error_data=error_data
    )

# Background tasks
def record_system_metrics_task():
    """Background task to record system metrics periodically."""
    with app.app_context():
        SystemMetric.record_metrics()

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Bad request'}), 400
    return render_template('errors/400.html'), 400

@app.errorhandler(401)
def unauthorized(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Unauthorized'}), 401
    return render_template('errors/401.html'), 401

@app.errorhandler(403)
def forbidden(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Forbidden'}), 403
    return render_template('errors/403.html'), 403

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('errors/404.html'), 404

@app.errorhandler(429)
def too_many_requests(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Too many requests'}), 429
    return render_template('errors/429.html'), 429

@app.errorhandler(500)
def internal_server_error(error):
    # Log the error
    logger.error(f"Internal server error: {error}")
    logger.error(traceback.format_exc())
    
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('errors/500.html'), 500

# Helper function to create admin user if none exists
def create_admin_user():
    """Create an admin user if no users exist in the database."""
    with app.app_context():
        if User.query.count() == 0:
            admin = User(
                username=app.config['ADMIN_USERNAME'],
                email=app.config['ADMIN_EMAIL'],
                role='admin'
            )
            admin.set_password(app.config['ADMIN_PASSWORD'])
            db.session.add(admin)
            db.session.commit()
            logger.info(f"Created admin user: {app.config['ADMIN_USERNAME']}")

# Initialize database and create tables
def init_db():
    """Initialize the database and create tables."""
    with app.app_context():
        db.create_all()
        logger.info("Database tables created")
        create_admin_user()

# Start background tasks
def start_background_tasks():
    """Start all background tasks."""
    # Start system metrics collection
    metrics_thread = threading.Thread(target=lambda: run_periodic_task(
        record_system_metrics_task, 
        interval=300  # every 5 minutes
    ))
    metrics_thread.daemon = True
    metrics_thread.start()

def run_periodic_task(task_func, interval):
    """Run a task periodically at the specified interval."""
    while True:
        try:
            task_func()
        except Exception as e:
            logger.error(f"Error in background task: {e}")
        time.sleep(interval)

# Main entry point
if __name__ == '__main__':
    # Initialize the database
    init_db()
    
    # Start background tasks
    start_background_tasks()
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000)