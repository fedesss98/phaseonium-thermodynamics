using Combinatorics
using TOML
using CSV
using DataFrames
using UUIDs


function read_csv_file(config)
    csv_file = "simulations/simulations_ledger.csv"
    csv = DataFrame()
    try
        csv = CSV.read(csv_file, DataFrame, header=[1,2]) 
    catch e
        println("CSV file error: $e")
    end
    if isempty(csv)
        df_headers = Dict{String, Any}()
        # Reconstruct the header
        for (category, subdict) in config
            if category != "variables"
                for (key, value) in subdict
                    df_headers["$(category)_$(key)"] = [category, key]
                end
            end
        end
        # Add the identifier
        df_headers["meta_name"] = ["meta", "name"]
        # Write the header
        CSV.write(csv_file, DataFrame(df_headers); writeheader=false)
    end
    return csv
end


function extract_parameter_ranges(config::Dict)
    # Initialize an empty dictionary to store parameter ranges
    parameter_ranges = Dict{String, Any}()
    
    for (category, sub_dict) in config
        for (key, value) in sub_dict
            # Store only parameters appearing in lists
            if value isa Vector
                # Create the full key path
                full_key = "$(category).$(key)"
                
                parameter_ranges[full_key] = value
            end
        end
    end
    
    return parameter_ranges
end


function config_exists(csv, config_row)
    if isempty(csv)
        return false
    end
    csv_to_compare = csv[:, Not(["meta_name", "meta_description"])]
    for row in eachrow(csv_to_compare)
        matches = []
        for param in names(row)
            if row[param] == config_row[param]
                push!(matches, true)
            else
                push!(matches, false)
            end
        end
        if all(matches)
            return true
        end
    end

    return false
end


function flatten_toml(uid, config::Dict)
    flat_dict = Dict{String, Any}()

    for (key, subdict) in config
        for (subkey, subvalue) in subdict
            flat_dict["$(key)_$(subkey)"] = subvalue
        end
    end
    # Add the identifier
    flat_dict["meta_name"] = uid

    return flat_dict
end


function update_config!(config, key, value)
    category, key = split(key, '.')
    if category != "variables"
        config[category][key] = value
    else
        # Look for values with that variable
        for (cat, sub_dict) in config
            for (sub_key, sub_value) in sub_dict
                if sub_value == ":$key"
                    # Update the value to the current variable value
                    config[cat][sub_key] = value
                end
            end
        end
    end
end


function update_csv!(csv, df_row)
    if isempty(csv)
        println("Creating!")
        csv = DataFrame(df_row)
    else
        println("Updating!")
        push!(csv, df_row)
    end
    return csv
end


function generate_configurations(dir="./"; config_file="")
    config = isempty(config_file) ? "/config.toml" : config_file
    try
        config = TOML.parsefile(dir * config)
    catch e
        println("Error reading configuration: $e")
    else
        println(config["meta"]["description"])
    end

    csv = read_csv_file(config)
    println("Simulations folder contains $(nrow(csv)) simulations.")
    # Extract repeated parameters
    params_ranges = extract_parameter_ranges(config)
    params_names = keys(params_ranges)
    params_values = values(params_ranges)
    params_combinations = collect(Iterators.product(params_values...))

    configs = DataFrame()
    skipped = 0
    for combination in params_combinations
        new_config = deepcopy(config)
        identifier = string(uuid4())[1:10]

        for (name, value) in zip(params_names, combination)
            update_config!(new_config, name, value)
        end
        # Remove the variables field from the config
        delete!(new_config, "variables")
        # Flatten the config dict as a DataFrame row 
        config_row = flatten_toml(identifier, new_config)

        # Check if this combination already exists
        if config_exists(csv, config_row)
            skipped += 1
            continue
        end
        
        # Create output directory
        mkpath("simulations/simulation_$identifier")
        # Save configuration file
        open("simulations/simulation_$identifier/config.toml", "w") do io
            TOML.print(io, new_config)
        end

        append!(configs, config_row)
    end

    println("$(skipped) existing configurations were skipped.")
    println("$(nrow(configs)) new configurations will be added.")

    # Save CSV if there are new configs
    if nrow(configs) > 0
        CSV.write("simulations/simulations_ledger.csv", configs; append=true)
    else
        println("No new configurations to add.")
    end

    return params_ranges
end