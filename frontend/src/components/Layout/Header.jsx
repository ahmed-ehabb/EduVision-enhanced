import './Header.css'

export default function Header({ title, subtitle, actions }) {
  return (
    <header className="header">
      <div className="header-content">
        <div className="header-text">
          {title && <h1 className="header-title">{title}</h1>}
          {subtitle && <p className="header-subtitle">{subtitle}</p>}
        </div>
        {actions && <div className="header-actions">{actions}</div>}
      </div>
    </header>
  )
}
