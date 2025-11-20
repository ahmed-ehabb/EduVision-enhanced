import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { AuthProvider } from './hooks/useAuth'
import ProtectedRoute from './components/Auth/ProtectedRoute'
import Login from './pages/Login'
import TeacherDashboard from './pages/TeacherDashboard'
import TeacherSessions from './pages/TeacherSessions'
import TeacherLectures from './pages/TeacherLectures'
import TeacherLectureUpload from './pages/TeacherLectureUpload'
import TeacherProcessingStatus from './pages/TeacherProcessingStatus'
import TeacherAnalytics from './pages/TeacherAnalytics'
import TeacherStudents from './pages/TeacherStudents'
import SystemTest from './pages/SystemTest'
import StudentDashboard from './pages/StudentDashboard'
import StudentSessions from './pages/StudentSessions'
import StudentStats from './pages/StudentStats'

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Toaster position="top-right" />
        <Routes>
          <Route path="/login" element={<Login />} />

          {/* Teacher Routes */}
          <Route
            path="/teacher/dashboard"
            element={
              <ProtectedRoute role="teacher">
                <TeacherDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/teacher/sessions"
            element={
              <ProtectedRoute role="teacher">
                <TeacherSessions />
              </ProtectedRoute>
            }
          />
          <Route
            path="/teacher/lectures"
            element={
              <ProtectedRoute role="teacher">
                <TeacherLectures />
              </ProtectedRoute>
            }
          />
          <Route
            path="/teacher/lectures/upload"
            element={
              <ProtectedRoute role="teacher">
                <TeacherLectureUpload />
              </ProtectedRoute>
            }
          />
          <Route
            path="/teacher/lectures/:jobId/status"
            element={
              <ProtectedRoute role="teacher">
                <TeacherProcessingStatus />
              </ProtectedRoute>
            }
          />
          <Route
            path="/teacher/analytics"
            element={
              <ProtectedRoute role="teacher">
                <TeacherAnalytics />
              </ProtectedRoute>
            }
          />
          <Route
            path="/teacher/students"
            element={
              <ProtectedRoute role="teacher">
                <TeacherStudents />
              </ProtectedRoute>
            }
          />
          <Route
            path="/teacher/test"
            element={
              <ProtectedRoute role="teacher">
                <SystemTest />
              </ProtectedRoute>
            }
          />

          {/* Student Routes */}
          <Route
            path="/student/dashboard"
            element={
              <ProtectedRoute role="student">
                <StudentDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/student/sessions"
            element={
              <ProtectedRoute role="student">
                <StudentSessions />
              </ProtectedRoute>
            }
          />
          <Route
            path="/student/stats"
            element={
              <ProtectedRoute role="student">
                <StudentStats />
              </ProtectedRoute>
            }
          />

          <Route path="/" element={<Navigate to="/login" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}

export default App
