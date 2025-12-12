# Chenhao Zhou - Academic Personal Website

🎓 **Live Site**: [chenhaozhou1996.github.io](https://chenhaozhou1996.github.io)

## Overview

This is a modern, responsive, and accessible academic personal website built with clean HTML/CSS and minimal JavaScript. The design emphasizes professional academic credibility, clean visual hierarchy, and optimal performance.

## 🌟 Features

### Design & UX
- **Modern Design System**: Unified color palette, typography, and component library
- **Fully Responsive**: Optimized for desktop, tablet, and mobile devices
- **Accessible**: WCAG 2.1 AA compliant with ARIA labels and semantic HTML
- **Performance Optimized**: Fast loading, minimal dependencies, no external frameworks
- **Professional Aesthetic**: Clean, academic-appropriate design

### Technical Features
- ✅ Semantic HTML5
- ✅ CSS Custom Properties (CSS Variables) for easy theming
- ✅ Smooth scroll navigation
- ✅ Back-to-top button
- ✅ Active navigation highlighting
- ✅ SEO optimized with Open Graph and Twitter Cards
- ✅ Schema.org structured data
- ✅ Print-friendly styles
- ✅ Sticky navigation header
- ✅ Responsive tables and cards

## 📁 File Structure

```
chenhaozhou1996.github.io/
├── index.html              # Homepage
├── research.html           # Research & Publications
├── cv.html                 # Curriculum Vitae (Online + PDF preview)
├── teaching.html           # Teaching Experience & Evaluations
├── conference.html         # Conferences & Presentations
├── milestones.html         # Research Milestones
├── professional.html       # Professional Background
├── styles/
│   └── main.css           # Unified stylesheet (design system)
├── images/
│   └── profile.jpg        # Profile photo
├── assets/
│   └── icons/             # Favicons (placeholder)
├── ml-course/             # Machine Learning Course Materials
├── *.pdf                  # CV, evaluations, and other documents
├── VERSION.txt            # Version tracking
└── README.md              # This file
```

## 🎨 Design System

### Color Palette

```css
--color-primary: #0f4c81        /* Deep Blue - Primary brand */
--color-primary-light: #1a5fa1  /* Lighter Blue - Hover states */
--color-primary-weak: #e6eef7   /* Very Light Blue - Backgrounds */
--color-text: #1f2937           /* Dark Gray - Primary text */
--color-text-muted: #6b7280     /* Medium Gray - Secondary text */
--color-bg: #f7f9fc             /* Light Blue-Gray - Page background */
--color-surface: #ffffff        /* White - Card backgrounds */
--color-border: #e5e7eb         /* Light Gray - Borders */
```

### Typography

- **Primary Font**: System UI fonts (ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial)
- **Fallback**: Sans-serif
- **Base Size**: 15px (0.9375rem)
- **Line Height**: 1.65 (relaxed for readability)

### Components

- **Cards**: Rounded corners (14px), subtle shadow, hover effects
- **Buttons**: Primary and outline variants with hover states
- **Badges**: Pill-shaped labels for status/categories
- **Tables**: Striped rows on hover, responsive design
- **Navigation**: Sticky header with backdrop blur

## 🔧 How to Update Content

### Adding a New Publication

**File**: `research.html`

1. Locate the appropriate section (Working Papers, Proceedings, etc.)
2. Add a new table row:

```html
<tr>
  <td><strong>Your Paper Title</strong></td>
  <td><span class="badge">Status</span> — Description</td>
  <td><em>Journal Name</em></td>
  <td>Authors</td>
</tr>
```

### Updating CV

**Option 1**: Replace PDF
- Upload new PDF as `CvResumeChenhao[DATE].pdf`
- Update filename reference in `cv.html` and `index.html`

**Option 2**: Update Online CV Content
- Edit the cards in `cv.html` directly

### Adding a Conference Presentation

**File**: `conference.html`

```html
<tr>
  <td>2025</td>
  <td>Conference Name — Track</td>
  <td>Presentation Topic</td>
  <td>Presenter/Chair</td>
  <td>Scheduled/Completed</td>
</tr>
```

### Updating Teaching Evaluations

**File**: `teaching.html`

1. Update the summary table with new scores
2. Add new comments to the "Representative Student Comments" section
3. Link to the new evaluation PDF if available

### Changing Colors/Theme

**File**: `styles/main.css` or inline `<style>` in each HTML file

Modify CSS custom properties:

```css
:root {
  --primary: #0f4c81;        /* Change primary color */
  --text: #1f2937;           /* Change text color */
  --bg: #f7f9fc;             /* Change background */
  /* etc. */
}
```

## 📱 Responsive Breakpoints

- **Desktop**: > 860px
- **Tablet**: 640px - 860px  
- **Mobile**: < 640px

All pages are fully responsive with appropriate adjustments at each breakpoint.

## ♿ Accessibility Features

- **Skip to Main Content** link for keyboard navigation
- **ARIA labels** on interactive elements
- **Semantic HTML5** elements (nav, main, header, footer, section)
- **Focus indicators** on all interactive elements
- **Alt text** on images
- **Sufficient color contrast** (WCAG AA)
- **Keyboard navigable** throughout

## 🚀 Performance Optimization

- **No external dependencies**: All CSS and JS inline or local
- **Minimal JavaScript**: ~50 lines total across all pages
- **Optimized images**: Profile photo is the only image on main pages
- **Efficient CSS**: Modern properties, no bloat
- **Fast loading**: Typical load time < 500ms

## 🔍 SEO Features

### Meta Tags
- Descriptive titles and meta descriptions on all pages
- Open Graph tags for social sharing
- Twitter Card tags
- Structured data (Schema.org JSON-LD) on homepage

### Best Practices
- Semantic HTML structure
- Proper heading hierarchy (h1 → h6)
- Descriptive link text
- Fast page load speeds
- Mobile-friendly design

## 🛠️ Development Workflow

### Testing Checklist

Before deploying changes:

1. **Cross-browser testing**
   - Chrome, Firefox, Safari, Edge
   
2. **Device testing**
   - Desktop (1920px, 1366px)
   - Tablet (768px, 1024px)
   - Mobile (375px, 414px)

3. **Accessibility validation**
   - [WAVE](https://wave.webaim.org/)
   - [axe DevTools](https://www.deque.com/axe/devtools/)
   
4. **Performance testing**
   - [Google PageSpeed Insights](https://pagespeed.web.dev/)
   - [Lighthouse](https://developers.google.com/web/tools/lighthouse)

5. **HTML/CSS validation**
   - [W3C HTML Validator](https://validator.w3.org/)
   - [W3C CSS Validator](https://jigsaw.w3.org/css-validator/)

### Deployment

This site is hosted on GitHub Pages. To deploy:

1. Make changes to HTML/CSS files
2. Commit and push to the `main` branch
3. Changes will be live within 1-2 minutes

```bash
git add .
git commit -m "Description of changes"
git push origin main
```

### Version Control

Update `VERSION.txt` after significant changes:

```
Site bundle: v[X] (root)
Built: YYYY-MM-DD HH:MM:SS UTC
```

## 📚 Common Tasks

### Add a New Page

1. Create `newpage.html` based on existing page template
2. Update navigation in ALL pages:

```html
<div class="menu">
  <!-- ... existing links ... -->
  <a href="newpage.html">New Page</a>
</div>
```

3. Ensure consistent styling with design system
4. Add appropriate meta tags and SEO elements

### Update Profile Photo

1. Replace `images/profile.jpg` with new image
2. Recommended size: 400x400px, optimized for web
3. Maintain aspect ratio (square)

### Add External Links (Google Scholar, ORCID, etc.)

Update footer in each page:

```html
<footer>
  <p style="margin-top: 8px;">
    <a href="https://scholar.google.com/..." target="_blank" rel="noopener">Google Scholar</a> ·
    <a href="https://orcid.org/..." target="_blank" rel="noopener">ORCID</a>
  </p>
</footer>
```

## 🎯 Future Enhancements

Potential improvements (not yet implemented):

- [ ] Publications filtering/sorting by year, type, or topic
- [ ] Citation export (BibTeX, APA, RIS formats)
- [ ] Image lazy loading for performance
- [ ] WebP image format support
- [ ] Contact form with validation
- [ ] Jekyll integration for easier content management
- [ ] Dark mode toggle
- [ ] Analytics integration (Google Analytics, Plausible, etc.)

## 📄 License

© 2025 Chenhao Zhou. All rights reserved.

## 📞 Contact

- **Email**: chenhao.zhou@rutgers.edu
- **LinkedIn**: [linkedin.com/in/chenhao-zhou-773499115](https://linkedin.com/in/chenhao-zhou-773499115)

---

**Last Updated**: January 2025  
**Version**: 10.0  
**Maintained by**: Chenhao Zhou
