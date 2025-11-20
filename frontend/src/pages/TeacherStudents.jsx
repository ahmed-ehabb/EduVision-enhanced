import Sidebar from '../components/Layout/Sidebar'
import Header from '../components/Layout/Header'

export default function TeacherStudents() {
  return (
    <div className="dashboard-layout">
      <Sidebar role="teacher" />
      <div className="dashboard-main">
        <Header title="Students" subtitle="View and manage students" />
        <div className="dashboard-content">
          <div className="card">
            <div className="card-body">
              <div className="empty-state">
                <h4>Student Management</h4>
                <p>Student management features coming soon.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
