import React, { useId } from "react";

const Input = React.forwardRef(function Input(
  { label, type = "text", className = "", ...props }: any,
  ref: React.Ref<HTMLInputElement>
) {
  const id = useId();

  return (
    <div className="w-full">
      {label && (
        <label
          htmlFor={id}
          className="inline-block mb-1 pl-1 text-sm font-medium text-gray-200"
        >
          {label}
        </label>
      )}
      <input
        id={id}
        ref={ref}
        type={type}
        {...props}
        className={`px-4 py-2 rounded-lg bg-gray-800 text-gray-200 outline-none 
          focus:bg-gray-700 focus:ring-2 focus:ring-green-500
          hover:bg-gray-700 transition duration-200 border border-gray-600 
          shadow-sm w-full ${className}`}
      />
    </div>
  );
});

export default Input;
