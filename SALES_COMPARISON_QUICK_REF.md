# Sales Comparison Route - Quick Reference

## 🚀 Quick Start

### Access the Route

```
http://localhost:3000/sales-comparison
```

### From MIS Dashboard

```html
<button onclick="window.location.href='http://localhost:3000/sales-comparison'">
  Talk with LLM
</button>
```

## 📋 Key Information

| Property             | Value                           |
| -------------------- | ------------------------------- |
| **Route**            | `/sales-comparison`             |
| **Page Context**     | `sales_comparison`              |
| **Database Table**   | `MIS.YEARLY_SALES_ANALYSIS_WEB` |
| **Theme Color**      | Emerald (#10b981)               |
| **Backend Endpoint** | `POST /chat`                    |

## 💬 Example Queries

```
✅ "Show monthly sales trend for 2024"
✅ "Compare sales year-over-year"
✅ "Show top performing months"
✅ "Analyze sales growth patterns"
✅ "What were Q1 sales in 2024?"
✅ "Compare Q1 vs Q2 sales"
```

## 🎯 Features

- ✅ Natural language queries
- ✅ AI-generated SQL
- ✅ Interactive charts (line/bar)
- ✅ Data tables
- ✅ PDF export
- ✅ SQL viewing/copying
- ✅ Follow-up suggestions
- ✅ Table access restriction

## 🔧 Technical Details

### Frontend

- **File**: `frontend/app/sales-comparison/page.tsx`
- **Framework**: Next.js 16 + React 19
- **UI**: Tailwind CSS + Recharts
- **Icons**: Lucide React

### Backend

- **File**: `backend/app.py`
- **Framework**: Flask + Python
- **Database**: Oracle 11g
- **LLM**: Groq (qwen-2.5-32b)

### API Call

```typescript
fetch("http://127.0.0.1:5000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message: "Your question here",
    page: "sales_comparison",
  }),
});
```

## 📁 Files

### Created

1. `frontend/app/sales-comparison/page.tsx` - Main component
2. `example-mis-dashboard.html` - Integration example
3. `SALES_COMPARISON_INTEGRATION.md` - Full documentation

### Referenced

1. `backend/app.py` - Backend configuration (line 80)
2. `frontend/app/page.tsx` - Original implementation

## 🎨 Customization

### Change Suggestions

Edit `page.tsx` line ~340:

```typescript
const suggested = ["Your custom suggestion 1", "Your custom suggestion 2"];
```

### Change Theme

Replace `emerald` with your color:

- `bg-emerald-50` → `bg-blue-50`
- `border-emerald-200` → `border-blue-200`
- `bg-emerald-600` → `bg-blue-600`

### Add More Tables

Edit `backend/app.py` line 80:

```python
"sales_comparison": [
    "MIS.YEARLY_SALES_ANALYSIS_WEB",
    "MIS.YOUR_NEW_TABLE",
],
```

## 🐛 Troubleshooting

| Issue               | Solution                                    |
| ------------------- | ------------------------------------------- |
| 404 Not Found       | Ensure Next.js dev server is running        |
| No data returned    | Check backend is running on port 5000       |
| Wrong table queried | Verify `PAGE_CONTEXT = 'sales_comparison'`  |
| SQL errors          | Check table exists and user has permissions |

## 📞 Support

1. Check browser console for errors
2. Review backend logs
3. Verify database connectivity
4. Check environment variables

## 🔗 Integration Methods

### HTML Button

```html
<button onclick="window.location.href='http://localhost:3000/sales-comparison'">
  Talk with LLM
</button>
```

### JavaScript Function

```javascript
function openAI() {
  window.location.href = "http://localhost:3000/sales-comparison";
}
```

### New Tab

```javascript
window.open("http://localhost:3000/sales-comparison", "_blank");
```

### Power BI

1. Add button visual
2. Action Type: Web URL
3. URL: `http://localhost:3000/sales-comparison`
4. Open in new tab: Yes

## 📊 Data Flow

```
User Question
    ↓
Frontend (/sales-comparison)
    ↓
Backend API (/chat)
    ↓
LLM (Groq)
    ↓
SQL Generation
    ↓
Oracle Database
    ↓
Results + Visualization
    ↓
User Dashboard
```

## ✅ Checklist

- [ ] Backend running on port 5000
- [ ] Frontend running on port 3000
- [ ] Database connection configured
- [ ] GROQ_API_KEY set in backend .env
- [ ] Table `MIS.YEARLY_SALES_ANALYSIS_WEB` exists
- [ ] User has SELECT permissions
- [ ] MIS Dashboard updated with redirect button

## 🚢 Production Deployment

### Build Frontend

```bash
cd frontend
npm run build
npm start
```

### Update MIS Dashboard

Change URL from:

```
http://localhost:3000/sales-comparison
```

To:

```
https://your-production-domain.com/sales-comparison
```

---

**Ready to use!** 🎉

For detailed documentation, see [SALES_COMPARISON_INTEGRATION.md](file:///c:/Users/sazzadsiddique/Documents/GitHub/incepta-vision/SALES_COMPARISON_INTEGRATION.md)
