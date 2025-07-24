import { Link, useNavigate } from "react-router-dom";
import { useForm } from "react-hook-form";
import Input from "../components/Input";
import Button from "../components/Button";
import { useAuth } from "../store/auth";
import API from "../lib/axios";

function Login() {
  const { handleSubmit, register } = useForm();
  const loginStore = useAuth((s) => s.login);
  const navigate = useNavigate();

  const onSubmit = async (data: any) => {
    try {
      const res = await API.post("/auth/login", data);
      const token = res.data.token;
      const user = res.data.user;

      localStorage.setItem("token", token);

      loginStore({
        _id: user._id,
        email: user.email,
        name: user.username,
      });

      // navigate("/upload-transcript");
      navigate("/analysis");
    } catch (e: any) {
      console.error("Login failed: ", e?.response?.data);
      alert("Login failed: " + e?.response?.data?.details);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900">
      <div className="w-full max-w-lg bg-gray-800 p-10 rounded-xl shadow-lg border border-gray-700">
        <h2 className="text-2xl font-extrabold text-center text-white">
          Login
        </h2>

        <p className="text-center text-sm text-gray-400 mt-4">
          Don't have an account?{" "}
          <Link
            to="/signup"
            className="text-green-400 hover:underline transition-all"
          >
            Sign Up
          </Link>
        </p>

        <form onSubmit={handleSubmit(onSubmit)} className="mt-8 space-y-5">
          <Input
            label="Email"
            placeholder="Enter your email"
            type="email"
            {...register("email", { required: true })}
          />
          <Input
            label="Password"
            placeholder="Enter your password"
            type="password"
            {...register("password", { required: true })}
          />
          <Button type="submit" className="w-full">
            Sign In
          </Button>
        </form>
      </div>
    </div>
  );
}

export default Login;
