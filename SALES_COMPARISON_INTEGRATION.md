# Sales Comparison Route - Integration Guide

## Overview

The `/sales-comparison` route provides a dedicated LLM-powered chat interface for analyzing data from the `MIS.YEARLY_SALES_ANALYSIS_WEB` database table. This route is designed to be accessed when users click "Talk with LLM" on the Sales Comparison section in the external MIS Dashboard.

## Features

### 🔒 Table Restriction

- **Hardcoded Page Context**: The route uses `sales_comparison` as the page context
- **Single Table Access**: Only queries the `MIS.YEARLY_SALES_ANALYSIS_WEB` table
- **Backend Enforcement**: The backend validates all SQL queries to ensure they only access allowed tables

### 🎨 Custom UI

- **Emerald Theme**: Uses emerald color scheme to distinguish from the main app
- **Sales-Specific Branding**: Clear indication that this is the "Sales Comparison Analysis" interface
- **Table Display**: Shows the database table name (`MIS.YEARLY_SALES_ANALYSIS_WEB`) for transparency
- **Custom Suggestions**: Pre-configured with sales comparison specific query suggestions

### 💬 Chat Interface

- **Natural Language Queries**: Ask questions in plain English
- **AI-Generated SQL**: Automatically generates Oracle SQL queries
- **Interactive Dashboard**: Visualize results with charts and tables
- **PDF Export**: Download reports as PDF documents
- **SQL Viewing**: Copy and review generated SQL queries

## URL Structure

### Dedicated Route

```
http://localhost:3000/sales-comparison
```

This route is self-contained and doesn't require any URL parameters.

### Alternative (Main App with Page Parameter)

```
http://localhost:3000/?page=sales_comparison
```

While this also works, the dedicated route is recommended for cleaner URLs and better user experience.

## Integration with MIS Dashboard

### HTML Example

```html
<button onclick="window.location.href='http://localhost:3000/sales-comparison'">
  Talk with LLM
</button>
```

### JavaScript Example

```javascript
function openSalesComparisonAI() {
  const aiAppUrl = "http://localhost:3000/sales-comparison";
  window.location.href = aiAppUrl;
  // Or use window.open(aiAppUrl, '_blank') to open in new tab
}
```

### Power BI Integration

If your MIS Dashboard is built with Power BI, you can add a button with a URL action:

1. Add a button visual to your Sales Comparison report page
2. Set the button text to "Talk with LLM"
3. Configure the button action:
   - Action Type: Web URL
   - URL: `http://localhost:3000/sales-comparison`
   - Open in new tab: Yes (recommended)

## Example Queries

The interface comes with pre-configured suggestions, but users can ask any question about the sales data:

### Monthly Trends

- "Show monthly sales trend for 2024"
- "What were the sales for each month in 2023?"
- "Compare monthly sales between 2023 and 2024"

### Year-over-Year Analysis

- "Compare sales year-over-year"
- "Show sales growth from last year to this year"
- "What is the percentage change in sales from 2023 to 2024?"

### Performance Analysis

- "Show top performing months"
- "Which months had the highest sales?"
- "Analyze sales growth patterns"

### Specific Queries

- "What were Q1 sales in 2024?"
- "Show sales for January to March 2024"
- "Compare Q1 vs Q2 sales"

## Technical Details

### Backend Configuration

The backend (`backend/app.py`) has the following configuration:

```python
PAGE_TABLES: Dict[str, List[str]] = {
    "sales_comparison": ["MIS.YEARLY_SALES_ANALYSIS_WEB"],
    # ... other page configurations
}
```

This ensures that when the page context is `sales_comparison`, only the `MIS.YEARLY_SALES_ANALYSIS_WEB` table can be queried.

### Frontend Implementation

The page component (`frontend/app/sales-comparison/page.tsx`) includes:

```typescript
// Hardcoded page context
const PAGE_CONTEXT = "sales_comparison";
const PAGE_TITLE = "Sales Comparison Analysis";
const TABLE_NAME = "MIS.YEARLY_SALES_ANALYSIS_WEB";
```

All API calls to the `/chat` endpoint automatically include this page context:

```typescript
body: JSON.stringify({ message: msgText, page: PAGE_CONTEXT });
```

## Security & Validation

### SQL Safety

- **Forbidden Keywords**: Blocks DROP, DELETE, UPDATE, INSERT, ALTER, etc.
- **Table Validation**: Verifies all referenced tables are in the allowed list
- **Row Limit**: Automatically limits results to 2000 rows to prevent performance issues

### Error Handling

- **Invalid Queries**: Returns helpful error messages
- **Database Errors**: Attempts to fix SQL errors automatically (up to 3 retries)
- **Connection Issues**: Displays user-friendly error messages

## Deployment

### Development

```bash
cd frontend
npm run dev
```

The app will be available at `http://localhost:3000/sales-comparison`

### Production

```bash
cd frontend
npm run build
npm start
```

Update the `AI_APP_BASE_URL` in your MIS Dashboard integration to point to your production URL:

```
https://your-domain.com/sales-comparison
```

## Customization

### Changing Suggestions

Edit the `suggested` array in `page.tsx`:

```typescript
const suggested = [
  "Your custom suggestion 1",
  "Your custom suggestion 2",
  "Your custom suggestion 3",
  "Your custom suggestion 4",
];
```

### Changing Theme Colors

The page uses emerald colors. To change:

1. Update background gradient: `bg-gradient-to-b from-emerald-50 to-white`
2. Update border colors: `border-emerald-200`
3. Update button colors: `bg-emerald-600 hover:bg-emerald-700`

### Adding More Tables

To allow additional tables, update the backend configuration in `app.py`:

```python
PAGE_TABLES: Dict[str, List[str]] = {
    "sales_comparison": [
        "MIS.YEARLY_SALES_ANALYSIS_WEB",
        "MIS.ADDITIONAL_TABLE_NAME",  # Add more tables here
    ],
}
```

## Troubleshooting

### Route Not Found (404)

- Ensure the Next.js dev server is running
- Verify the file exists at `frontend/app/sales-comparison/page.tsx`
- Check for TypeScript compilation errors

### No Data Returned

- Verify the backend is running on port 5000
- Check the `NEXT_PUBLIC_API_BASE` environment variable
- Ensure database connection is configured correctly in backend `.env`

### Wrong Table Being Queried

- Verify the `PAGE_CONTEXT` constant is set to `'sales_comparison'`
- Check backend logs to see which page context is being received
- Ensure the backend `PAGE_TABLES` configuration is correct

### SQL Errors

- Check if the table `MIS.YEARLY_SALES_ANALYSIS_WEB` exists in your database
- Verify database user has SELECT permissions on the table
- Review the generated SQL in the chat interface

## Support

For issues or questions:

1. Check the browser console for JavaScript errors
2. Review backend logs for API errors
3. Verify database connectivity
4. Ensure all environment variables are set correctly

## Example Files

- **MIS Dashboard Integration**: See `example-mis-dashboard.html` for a complete example
- **Backend Configuration**: See `backend/app.py` lines 78-147 for page table mappings
- **Frontend Component**: See `frontend/app/sales-comparison/page.tsx` for the implementation
