import React, { createContext, useContext, useState, useEffect } from "react";

const AuthContext = createContext(null);

const STORAGE_KEY = "screening_auth_user";

// Uses sessionStorage (not localStorage) on purpose: sessionStorage clears
// automatically when the browser tab/window is closed, so every fresh visit
// to the site requires logging in again. Refreshing the SAME tab while it's
// still open keeps you logged in (this is normal browser behavior — a full
// re-login on every single refresh would also log out mid-review, which
// would be disruptive for an officer partway through a screening).
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = sessionStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        setUser(JSON.parse(stored));
      } catch {
        sessionStorage.removeItem(STORAGE_KEY);
      }
    }
    setLoading(false);
  }, []);

  function login(name, role) {
    const userObj = { name, role };
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(userObj));
    setUser(userObj);
  }

  function logout() {
    sessionStorage.removeItem(STORAGE_KEY);
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
