for file in *
    mv $file $(echo $file | cut -c 15-）
end

