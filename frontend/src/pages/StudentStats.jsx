import Sidebar from '../components/Layout/Sidebar'
import Header from '../components/Layout/Header'

export default function StudentStats() {
  return (
    <div className="dashboard-layout">
      <Sidebar role="student" />
      <div className="dashboard-main">
        <Header title="My Stats" subtitle="View your engagement statistics" />
        <div className="dashboard-content">
          <div className="card">
            <div className="card-body">
              <div className="empty-state">
                <h4>Personal Statistics</h4>
                <p>Your engagement statistics and insights will appear here.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
