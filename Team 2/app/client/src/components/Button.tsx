import { type FC, type ButtonHTMLAttributes } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  Icon?: FC<{ size?: number }>;
};

const Button: FC<ButtonProps> = ({
  className = "",
  children,
  Icon,
  ...props
}) => {
  return (
    <button
      {...props}
      className={`flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg transition duration-200 ${className}`}
    >
      {Icon && <Icon size={18} />}
      {children}
    </button>
  );
};

export default Button;
