# Incepta Vision - AI-Powered MIS Dashboard

> Enterprise-grade AI assistant for intelligent data analysis and visualization of Management Information Systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)

## 🌟 Features

- **Natural Language Queries**: Ask questions in plain English, get instant SQL-powered insights
- **AI-Generated Visualizations**: Automatic chart selection and dashboard generation
- **Multi-Table Support**: Secure, page-specific table access control
- **PDF Export**: Professional report generation with charts and data tables
- **Real-time Analytics**: Fast query execution with Oracle database integration
- **Responsive Design**: Beautiful, modern UI that works on all devices

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Next.js 16    │ ───▶ │   Flask Backend  │ ───▶ │  Oracle 11g DB  │
│   Frontend      │      │   + Groq LLM     │      │   (MIS Schema)  │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

### Tech Stack

**Frontend:**

- Next.js 16 (React 19)
- TypeScript
- Tailwind CSS
- Recharts for visualizations
- Marked for markdown rendering

**Backend:**

- Python 3.11+
- Flask web framework
- Groq LLM (qwen-2.5-32b)
- Oracle 11g database
- Pydantic for validation

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Node.js 20 or higher
- Oracle 11g database access
- Groq API key ([Get one here](https://console.groq.com))

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run development server
python app.py
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local if needed

# Run development server
npm run dev
```

Visit `http://localhost:3000` to see the application.

## 📖 Usage

### Sales Comparison Route

Navigate to `/sales-comparison` for dedicated sales analysis:

```
http://localhost:3000/sales-comparison
```

**Example Queries:**

- "Show monthly sales trend for 2024"
- "Compare depot vs export sales"
- "What was the growth rate in March?"
- "Show year-over-year comparison"

### Integration with MIS Dashboard

Add a "Talk with LLM" button to your existing dashboard:

```html
<button onclick="window.location.href='http://localhost:3000/sales-comparison'">
  Talk with LLM
</button>
```

See [SALES_COMPARISON_INTEGRATION.md](SALES_COMPARISON_INTEGRATION.md) for detailed integration guide.

## 🔧 Configuration

### Backend Environment Variables

```env
# Database
DB_HOST=your-oracle-host
DB_PORT=1521
DB_SERVICE=your-service-name
DB_USER=your-username
DB_PASSWORD=your-password

# LLM
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=qwen-2.5-32b

# Optional
ORACLE_HOME=/path/to/oracle/instantclient
```

### Frontend Environment Variables

```env
NEXT_PUBLIC_API_BASE=http://localhost:5000
```

## 📁 Project Structure

```
incepta-vision/
├── backend/
│   ├── config.py              # Configuration management
│   ├── models.py              # Pydantic models
│   ├── app.py                 # Main Flask application
│   ├── services/              # Business logic layer
│   │   ├── database.py        # Database operations
│   │   └── sql_service.py     # SQL generation & validation
│   ├── dictionary.json        # Table schema definitions
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── app/                   # Next.js app directory
│   │   ├── page.tsx           # Main chat page
│   │   └── sales-comparison/  # Sales comparison route
│   ├── components/            # Reusable React components
│   │   ├── ChatMessage.tsx    # Chat message component
│   │   └── ErrorBoundary.tsx  # Error handling
│   ├── hooks/                 # Custom React hooks
│   │   └── useChat.ts         # Chat functionality hook
│   ├── lib/                   # Utilities and helpers
│   │   ├── api.ts             # API client
│   │   ├── types.ts           # TypeScript types
│   │   └── utils.ts           # Utility functions
│   └── package.json           # Node dependencies
└── README.md                  # This file
```

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest --cov=. --cov-report=html
```

### Frontend Tests

```bash
cd frontend
npm test
```

## 📦 Deployment

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Manual Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## 🔒 Security

- SQL injection prevention through parameterized queries
- Forbidden keyword blocking (DROP, DELETE, etc.)
- Table access control per page context
- Automatic row limiting (2000 max)
- Environment variable protection

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Groq for providing fast LLM inference
- Oracle for database technology
- Next.js and React teams for excellent frameworks

## 📞 Support

For issues and questions:

- Create an issue on GitHub
- Check existing documentation
- Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Built with ❤️ for Incepta Pharmaceuticals**
