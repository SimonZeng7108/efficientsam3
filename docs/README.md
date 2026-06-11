# GitHub Pages

This folder contains the source for the EfficientSAM3 GitHub Pages site.

## Setup

The site is built with Jekyll and deployed via GitHub Pages.

## Structure

```
docs/
├── _config.yml          # Jekyll configuration
├── _layouts/           # Page layouts
├── _includes/          # Reusable components
├── assets/             # CSS, images, JS
├── index.md           # Home page
├── models.md          # Model zoo
└── documentation.md   # Documentation
```

## Local Development

To preview locally:

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Visit http://localhost:4000/efficientsam3/

## Deployment

Push to the `main` branch. GitHub Pages will automatically build and deploy.

The site will be available at: https://simonzeng7108.github.io/efficientsam3/

## Enabling GitHub Pages

1. Go to repository Settings > Pages
2. Select Source: Deploy from a branch
3. Select Branch: main, Folder: /docs
4. Click Save
