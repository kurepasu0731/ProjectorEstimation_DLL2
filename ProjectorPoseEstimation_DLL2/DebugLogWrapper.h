#ifndef DEBUGLOGWRAPPER_H
#define DEBUGLOGWRAPPER_H

#pragma once

#include <string>

#define DLLExport __declspec (dllexport)

extern "C"
{
	typedef void (*debug_log_func_type)(std::string);

	debug_log_func_type debug_log_func;

	int counter = 0;
        
	void debug_log(std::string msg)
    {
        if (debug_log_func != nullptr) debug_log_func(msg);
    }
    
    DLLExport void set_debug_log_func(debug_log_func_type func)
    {
        debug_log_func = func;
		counter = 0;
    }
    
    DLLExport void debug_log_test()
    {
		std::string log = "testLog: " + std::to_string(counter);
        debug_log(log);
		counter++;
    }
}
#endif