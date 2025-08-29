def check_and_update_quota(tokens_used):
    """
    Checks if performing an action would exceed quotas and updates usage.

    Args:
        tokens_used: The number of tokens the current action would use.

    Returns:
        True if the action is within quotas, False otherwise, sleeps if necessary.
    """
    global requests_this_minute, tokens_this_minute, requests_today
    global start_time_minute, start_time_day

    current_time = time.time()

    # Calculate time elapsed in the current minute and day
    time_elapsed_this_minute = current_time - start_time_minute
    time_elapsed_today = current_time - start_time_day

    # Reset minute counts if a minute has passed
    if time_elapsed_this_minute >= 60:
        requests_this_minute = 0
        tokens_this_minute = 0
        start_time_minute = current_time
        time_elapsed_this_minute = 0 # Reset elapsed time for the new minute

    # Reset daily counts if a day has passed (86400 seconds in a day)
    if time_elapsed_today >= 86400:
        requests_today = 0
        start_time_day = current_time
        time_elapsed_today = 0 # Reset elapsed time for the new day


    # Check if RPD limit is reached
    if requests_today >= RPD_LIMIT:
        print("RPD limit exceeded. Cannot make more requests today.")
        return False

    # Calculate time needed before the next request based on RPM and TPM
    # Ensure we don't divide by zero if limits are zero
    time_needed_rpm = 0
    if RPM_LIMIT > 0:
        # Calculate remaining capacity for requests in the current minute
        remaining_requests_in_minute = RPM_LIMIT - requests_this_minute - 1
        if remaining_requests_in_minute < 0:
             # If adding this request exceeds RPM, calculate time until next minute reset
             time_needed_rpm = 60 - time_elapsed_this_minute


    time_needed_tpm = 0
    if TPM_LIMIT > 0 and tokens_used > 0:
        # Calculate remaining capacity for tokens in the current minute
        remaining_tokens_in_minute = TPM_LIMIT - tokens_this_minute - tokens_used
        if remaining_tokens_in_minute < 0:
            # If adding these tokens exceeds TPM, calculate time until next minute reset
             time_needed_tpm = 60 - time_elapsed_this_minute

    # Determine the maximum time needed based on both limits and remaining time in the minute
    sleep_duration = max(time_needed_rpm, time_needed_tpm)

    # Ensure sleep duration is non-negative
    sleep_duration = max(0, sleep_duration)


    if sleep_duration > 0:
        print(f"Quota limit approaching. Sleeping for {sleep_duration:.2f} seconds to stay within limits.")
        time.sleep(sleep_duration)
        # After sleeping, update the current time and elapsed time for the minute
        current_time = time.time()
        start_time_minute = current_time # Reset start time for the new minute after sleeping
        time_elapsed_this_minute = 0
        requests_this_minute = 0 # Reset counts after sleeping for a new minute
        tokens_this_minute = 0


    # If within minute quotas and RPD, update usage
    requests_this_minute += 1
    tokens_this_minute += tokens_used
    requests_today += 1
    return True